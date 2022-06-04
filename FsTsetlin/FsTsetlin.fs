namespace FsTsetlin 
open TorchSharp
open System
open MBrace.FsPickler

type Config = 
    {
        s                           : float32
        T                           : float32
        TAStates                    : int
        dtype                       : torch.ScalarType
        Device                      : torch.Device
        ClausesPerClass             : int
        InputSize                   : int
        Classes                     : int // 2 or more
        BoostTruePositiveFeedback   : bool
    }
    with 
        static member Default  = 
            {
                s                           = 3.9f
                T                           = 15.0f
                TAStates                    = 100
                dtype                       = torch.int32
                Device                      = torch.CPU
                ClausesPerClass             = 100
                InputSize                   = 0
                Classes                     = 2
                BoostTruePositiveFeedback   = false
            }

/// cache invariates for optimization
type TMState = 
    {
        Config          : Config
        PolarityIndex   : torch.Tensor
        PolaritySign    : torch.Tensor
        PlrtySignClass  : torch.Tensor
        PayoutMatrix    : torch.Tensor
        MidState        : torch.Tensor
        LowState        : torch.Tensor
        HighState       : torch.Tensor
        Ones            : torch.Tensor
        Zeros           : torch.Tensor
        TPlus           : torch.Tensor
        TMinus          : torch.Tensor
        T2              : torch.Tensor
        Y1              : torch.Tensor
        Y0              : torch.Tensor
        IsBinary        : bool
        ChunkedClauses  : torch.Tensor[]
        ChunkedWeights  : torch.Tensor[]
    }

type TM =
    {
        Clauses         : torch.Tensor
        Weights         : torch.Tensor
        TMState         : TMState
    }

        
module Eval = 
    
    ///eval each literal against each tsetlin automaton (TA) in the clauses
    let evalTA tmst trainMode (clauses:torch.Tensor)  (input:torch.Tensor) = 
        use input2   = input.broadcast_to(clauses.shape)                //make input the same shape as clauses
        let actions  = clauses.greater(tmst.MidState)                 //determine 'include'/'exclude' actions
        let taOutput = torch.where(actions,input2,tmst.Ones)          //take input if action = include else 1 (which skips excluded input when and'ing)
        let out = 
            if not trainMode then   
                use oneInc = actions.any(1L,keepDim=true)               //true if there is at least 1 'include' per clause
                if oneInc.any().ToBoolean() then
                    let adjOutput = torch.where(oneInc,taOutput,tmst.Zeros)
                    taOutput.Dispose()
                    adjOutput
                else
                    taOutput
            else
                taOutput
        actions,out

    ///AND the outputs of TAs by clause
    let andClause tmst (evals:torch.Tensor)  =
        use fltr = evals.bool()
        use fltr2 = fltr.all(dimension=1L)
        torch.where(fltr2,tmst.Ones,tmst.Zeros)

    /////sum positive and negative polarity clause outputs
    //let sumClauses tmst (clauseEvals:torch.Tensor) =
    //    use withPlry = clauseEvals.mul(tmst.PolaritySign)
    //    withPlry.sum(``type``=torch.float32)

    ///sum positive and negative polarity clause outputs for a single class
    let sumClass tmst (clauseEvals:torch.Tensor) (weights:torch.Tensor) =
        use wtd = clauseEvals.mul(weights)
        use withPlry = wtd.mul(tmst.PlrtySignClass)
        withPlry.sum(``type``=torch.float32)

    /////sum positive and negative polarity clause outputs per class for all classes
    let sumClausesMulticlass tmst (clauseEvals:torch.Tensor) (weights:torch.Tensor) =
        use wts = weights.reshape(1L,-1L)
        use wtd = clauseEvals.mul(wts)
        use withPlry = wtd.mul(tmst.PolaritySign)
        use byClass = withPlry.reshape( int64 tmst.Config.Classes, -1L )
        byClass.sum(1L, ``type``= torch.float32)

module Train = 
    ///obtain +/- reward probabilities for each TA (for type I and II feedback)
    let rewardProb tmst (actions:torch.Tensor) (clauseEvals:torch.Tensor) (X:torch.Tensor,y:torch.Tensor) =
        use ce_t = clauseEvals.reshape(-1L,1L)
        (*polarity    literal     action  Cw  y   ->  p_reward *)
        use literal_f1 = X.expand_as(actions)
        use literal_f2 = literal_f1.reshape([|1L;-1L|])
        use literal_f  = literal_f2.to_type(torch.int64)
        use action_f1 = torch.where(actions,tmst.Ones,tmst.Zeros)
        use action_f2 = action_f1.reshape([|1L;-1L|])
        use action_f  = action_f2.to_type(torch.int64)
        use cw_f1 = ce_t.broadcast_to(actions.shape)
        use cw_f2 = cw_f1.reshape([|1L;-1L|])
        use cw_f  = cw_f2.to_type(torch.int64)
        use y_f1 = y.expand_as(actions)
        use y_f2 = y_f1.reshape([|1L;-1L|])
        use y_f  = y_f2.to_type(torch.int64)
        use plrt_f = tmst.PolarityIndex.reshape([|1L;-1L|])
        let pReward = tmst.PayoutMatrix.index([|plrt_f; literal_f; action_f; cw_f; y_f|])
        pReward

    ///postive, negative or no feedback for each TA
    let taFeeback tmst (feedbackFilter:torch.Tensor) (v:torch.Tensor) (pReward:torch.Tensor) (y:torch.Tensor) =
        use pr1 = pReward.reshape(feedbackFilter.shape.[0], -1L)
        use ff1 =  feedbackFilter.reshape(-1L,1L)
        use ff2 = ff1.expand_as(pr1)  
        use feedbackExpanded = ff2.reshape(1L,-1L)
        use uRandRwrd = torch.rand_like(pReward)         //tensor of uniform random values for Reward/Penalty selection
        use pRwrd_abs = pReward.abs()
        use pSelFltr = uRandRwrd.less_equal(pRwrd_abs)
        use pFilter = feedbackExpanded.logical_and(pSelFltr)
        use rwdSign = pReward.sign()
        use rwdSignInt = rwdSign.to_type(tmst.Config.dtype)
        let fb = rwdSignInt.where(pFilter,tmst.Zeros)
        fb

    ///calculate feedback incr/decr values based on selected feedback reward(+)/penalty(-)/ignore(0) and TA state
    let feedbackIncrDecr tmst (actions:torch.Tensor) (feedback:torch.Tensor) =
        use one = torch.tensor([|1|],dtype=tmst.Config.dtype, device=tmst.Config.Device)
        use minusOne = torch.tensor([|-1|],dtype=tmst.Config.dtype, device=tmst.Config.Device)
        use filterBin = torch.where(actions,one,minusOne)
        let fp = filterBin.mul(feedback.reshape(actions.shape))
        fp

    ///inplace update clauses
    let updateClauses_ tmst (clauses:torch.Tensor) (incrDecr:torch.Tensor) =
        clauses.add_(incrDecr).clamp_(tmst.LowState,tmst.HighState)

    ///udpated clauses as a new tensor (for debugging purposes)
    let updateClauses tmst (clauses:torch.Tensor) (incrDecr:torch.Tensor) =
        let clss = clauses.add(incrDecr)
        clss.clamp_(tmst.LowState,tmst.HighState)

    ///update the clauses for a single class - used in multi-class scenario
    let updateClass tmst actions clauses clauseEvals (v:torch.Tensor) (X,y:torch.Tensor) = 
        use uRand = torch.rand_like(clauseEvals,dtype=torch.ScalarType.Float16)              //tensor of uniform random values for Feedback selection
        use feebackFilter =                                   //bool tensor; true if random <= [selY0 or selY1] (based on the value of y )
            if y.ToSingle() <= 0.f then 
                use selY0 = (tmst.TPlus + v.clamp(tmst.TMinus,tmst.TPlus)) / tmst.T2  //feedback selection prob. when y=0 
                uRand.less_equal(selY0) 
            else 
                use selY1 = (tmst.TPlus - v.clamp(tmst.TMinus,tmst.TPlus)) / tmst.T2  //feedback selection prob. when y=1
                uRand.less_equal(selY1)              
        use pReward = rewardProb tmst actions clauseEvals (X,y)
        use feedback = taFeeback tmst feebackFilter v pReward y
        use fbIncrDecr = feedbackIncrDecr tmst actions feedback 
        updateClauses_ tmst clauses fbIncrDecr |> ignore

    let trainStepMulticlass tmst (clauses:torch.Tensor) (X,y:torch.Tensor) = 
        let idx1 = y.ToInt64()
        use notY = 
            if tmst.IsBinary then
                let notYIdx = if idx1 = 1L then 0L else 1L            
                torch.tensor([|notYIdx|],dtype=torch.int64,device=tmst.Config.Device)
            else
                let remClss = [|for i in 0L .. int64 (tmst.Config.Classes - 1) do if i <> idx1 then yield i |] 
                use tRemClss = torch.tensor(remClss,dtype=torch.int64,device=tmst.Config.Device)
                use notYIdx = torch.randint(int tRemClss.shape.[0],[|1|],dtype=torch.int64,device=tmst.Config.Device)
                tRemClss.[notYIdx]        
        //when class = y
        let yIdx = y.ToInt32()
        let v1Clauses = tmst.ChunkedClauses.[yIdx]
        let v1Actions,v1TAEvals = Eval.evalTA tmst true v1Clauses X
        use v1Actions = v1Actions
        use v1TAEvals = v1TAEvals
        use v1ClsEvals = Eval.andClause tmst v1TAEvals
        let w1        = tmst.ChunkedWeights.[yIdx]
        use v1        = Eval.sumClass tmst v1ClsEvals w1
        updateClass tmst v1Actions v1Clauses v1ClsEvals v1 (X,tmst.Y1)
        //when class not y (randomly chosen when classes > 2)
        let notYIdx = notY.ToInt32()
        let v0Clauses = tmst.ChunkedClauses.[notYIdx]
        let v0Actions,v0TAEvals = Eval.evalTA tmst true v0Clauses X
        let v0Actions = v0Actions
        let v0TAEvals = v0TAEvals
        use v0ClsEvals = Eval.andClause tmst v0TAEvals
        let w0        = tmst.ChunkedWeights.[notYIdx]
        use v0        = Eval.sumClass tmst v0ClsEvals w0
        updateClass tmst v0Actions v0Clauses v0ClsEvals v0 (X,tmst.Y0)        

    /////update clauses on single input - optimized for producton
    //let trainStep tmst clauses (X,y) = 
    //    let actions,taEvals = Eval.evalTA tmst true clauses X //num_clauses * input
    //    use actions = actions
    //    use taEvals = taEvals
    //    use clauseEvals = Eval.andClause tmst taEvals
    //    use v = Eval.sumClauses tmst clauseEvals
    //    use pReward = rewardProb tmst actions clauseEvals (X,y)
    //    use feedback = taFeeback tmst v pReward y
    //    use fbIncrDecr = feedbackIncrDecr tmst actions feedback 
    //    updateClauses_ tmst clauses fbIncrDecr |> ignore

    ////debug version of update that returns intermediate results
    //let trainStepDbg tmst clauses (X,y)  =
    //    let actions,taEvals = Eval.evalTA tmst true clauses X //num_clauses * input
    //    let clauseEvals = Eval.andClause tmst taEvals
    //    let v = Eval.sumClauses tmst clauseEvals
    //    let pReward = rewardProb tmst actions clauseEvals (X,y)
    //    let feedback = taFeeback tmst v pReward y
    //    let fbIncrDecr = feedbackIncrDecr tmst actions feedback 
    //    let updtClss = updateClauses tmst clauses fbIncrDecr 
    //    taEvals,clauseEvals,v,pReward,feedback,fbIncrDecr,updtClss

module TM =
    let inaction = 0.0f
    let pnlty    = -1.0f

    let payout boostTPF s = 
        let ``1/s``     = 1.0f / s
        let ``(s-1)/s`` = (s - 1.0f) / s
        let ``1 or (s-1)/s``  = if boostTPF then 1.0f else ``(s-1)/s``
        [|
        (*polarity    literal     action  Cw  y     p_reward *)
        (*0           0           0       0   0 *)  ``1/s``             //0
        (*0           0           0       0   1 *)  inaction            //1
        (*0           0           0       1   0 *)  ``1/s``             //2
        (*0           0           0       1   1 *)  pnlty               //3
        (*0           0           1       0   0 *)  -``1/s``            //4
        (*0           0           1       0   1 *)  inaction            //5
        (*0           0           1       1   0 *)  inaction            //6
        (*0           0           1       1   1 *)  inaction            //7
        (*0           1           0       0   0 *)  ``1/s``             //8
        (*0           1           0       0   1 *)  inaction            //9
        (*0           1           0       1   0 *)  -``(s-1)/s``        //10
        (*0           1           0       1   1 *)  inaction            //11
        (*0           1           1       0   0 *)  -``1/s``            //12
        (*0           1           1       0   1 *)  inaction            //13
        (*0           1           1       1   0 *)  ``(s-1)/s``         //14
        (*0           1           1       1   1 *)  inaction            //15
        (*1           0           0       0   0 *)  inaction            //16
        (*1           0           0       0   1 *)  ``1/s``             //17
        (*1           0           0       1   0 *)  pnlty               //18
        (*1           0           0       1   1 *)  ``1/s``             //19
        (*1           0           1       0   0 *)  inaction            //20
        (*1           0           1       0   1 *)  -``1/s``            //21
        (*1           0           1       1   0 *)  inaction            //22
        (*1           0           1       1   1 *)  inaction            //23
        (*1           1           0       0   0 *)  inaction            //24
        (*1           1           0       0   1 *)  ``1/s``             //25
        (*1           1           0       1   0 *)  inaction            //26
        (*1           1           0       1   1 *)  -``1 or (s-1)/s``   //27
        (*1           1           1       0   0 *)  inaction            //28
        (*1           1           1       0   1 *)  -``1/s``            //29
        (*1           1           1       1   0 *)  inaction            //30
        (*1           1           1       1   1 *)  ``1 or (s-1)/s``    //31
        |]

    let create (cfg:Config) =
        if cfg.ClausesPerClass % 2 <> 0 then failwithf "clauses per class should be an even number (for negative/positive polarity)"
        let rng = System.Random()
        let isBinary = cfg.Classes = 2
        let numTAs = cfg.Classes * cfg.ClausesPerClass * (cfg.InputSize * 2)
        let initialState = [|for i in 1..numTAs -> if rng.NextDouble() < 0.5 then cfg.TAStates else cfg.TAStates+1|]
        let clauses = torch.tensor(initialState, dtype=cfg.dtype, dimensions = [|int64 (cfg.Classes * cfg.ClausesPerClass); int64 (cfg.InputSize * 2) |], device=cfg.Device)
        let plrtySgn = [|for i in 1 .. (cfg.Classes * cfg.ClausesPerClass) -> if i % 2 = 0 then -1 else 1|]  //polarity sign +1/-1; used for summing clause evaluation
        let plrtySgnCls = [|for i in 1 .. cfg.ClausesPerClass -> if i % 2 = 0 then -1 else 1|]
        let numClauses = (*if isBinary then cfg.Classes * cfg.ClausesPerClass else *)cfg.ClausesPerClass
        let plrtyBin = [|for i in 1 .. numClauses -> i % 2|]                                                  //polarity index 1/0; used for indexing into the payout matrix
        let polarityIdx = 
            //if isBinary then
            //    let polarity = torch.tensor(plrtyBin, dtype=torch.int64, device=cfg.Device, dimensions=[|clauses.shape.[0]; 1L|])
            //    polarity.expand_as(clauses).reshape([|1L;-1L|])
            //else
                let polarity = torch.tensor(plrtyBin, dtype=torch.int64, device=cfg.Device, dimensions=[|int64 cfg.ClausesPerClass; 1L|])
                polarity.broadcast_to(int64 cfg.ClausesPerClass, int64 (2 * cfg.InputSize))
        let tPlus = torch.tensor(cfg.T, device=cfg.Device)
        let tMinus = torch.tensor(-cfg.T, device=cfg.Device)
        let t2 = torch.tensor(2.0f * cfg.T,device=cfg.Device)
        let ones  = torch.tensor([|1|], dtype = cfg.dtype, device = cfg.Device)
        let zeros = torch.tensor([|0|], dtype = cfg.dtype, device = cfg.Device)
        let payout = payout cfg.BoostTruePositiveFeedback cfg.s
        let weights = torch.tensor([|1|], dtype = cfg.dtype, device = cfg.Device).broadcast_to([| int64 cfg.Classes; int64 cfg.ClausesPerClass |])
        let chunkedClauses = clauses.chunk(int64 cfg.Classes)
        let chunkedWts = weights.chunk(int64 cfg.Classes)
        let tmState =
            {
                PolarityIndex   = polarityIdx
                PolaritySign    = torch.tensor(plrtySgn, dtype=cfg.dtype, device=cfg.Device)
                PlrtySignClass  = torch.tensor(plrtySgnCls, dtype=cfg.dtype, device=cfg.Device)
                PayoutMatrix    = torch.tensor(payout, dimensions = [|2L;2L;2L;2L;2L|], device=cfg.Device)   
                MidState        = torch.tensor([|0|],dtype=cfg.dtype, device=cfg.Device)
                LowState        = torch.tensor([|-cfg.TAStates|],dtype=cfg.dtype,device=cfg.Device)
                HighState       = torch.tensor([|cfg.TAStates|],dtype=cfg.dtype,device=cfg.Device)
                Zeros           = zeros
                Ones            = ones
                Config          = cfg
                TPlus           = tPlus
                TMinus          = tMinus
                T2              = t2
                Y1              = 1L.ToTensor(device=cfg.Device)
                Y0              = 0L.ToTensor(device=cfg.Device)
                IsBinary        = isBinary
                ChunkedClauses  = chunkedClauses
                ChunkedWeights  = chunkedWts
            }
        {
            Clauses     = clauses
            Weights     = weights
            TMState  = tmState
        }

    let train (X,y) (tm:TM) =
        //if tm.TMState.IsBinary then
        //    Train.trainStep tm.TMState tm.Clauses (X,y)
        //else
            Train.trainStepMulticlass tm.TMState tm.Clauses (X,y)

    let trainBatch (X:torch.Tensor,y:torch.Tensor) (tm:TM) =
        let batchSze = X.shape.[0]
        for i in 0L .. batchSze - 1L do
            train (X.[i],y.[i]) tm

    let predict X (tm:TM) =
        let actions,taEvals = Eval.evalTA tm.TMState false tm.Clauses X //num_clauses * input
        use actions = actions
        use taEvals = taEvals
        use clauseEvals = Eval.andClause tm.TMState taEvals
        use byClassSum = Eval.sumClausesMulticlass tm.TMState clauseEvals tm.Weights
        use idx = byClassSum.argmax()
        idx.ToInt32()

    let predictBatch (X:torch.Tensor) (tm:TM) =
        let batchSze = X.shape.[0]
        [|for i in 0L .. batchSze - 1L do
            predict X.[i] tm
        |]

    type private State = 
        {
            s                           : float32
            T                           : float32
            TAStates                    : int
            dtype                       : string
            ClausesPerClass             : int
            InputSize                   : int
            Classes                     : int // 2 or more            
            BoostTruePositiveFeedback   : bool
        }

    let private toState (cfg:Config) = 
        {
            s                           = cfg.s
            T                           = cfg.T
            BoostTruePositiveFeedback   = cfg.BoostTruePositiveFeedback
            TAStates                    = cfg.TAStates
            dtype                       = cfg.dtype.ToString()
            ClausesPerClass             = cfg.ClausesPerClass
            InputSize                   = cfg.InputSize
            Classes                     = cfg.Classes
        }

    let private toConfig dvc (cfg:State) : Config = 
        let dt : torch.ScalarType = System.Enum.Parse<torch.ScalarType>(cfg.dtype)
        {
            s                           = cfg.s
            T                           = cfg.T
            BoostTruePositiveFeedback   = cfg.BoostTruePositiveFeedback
            TAStates                    = cfg.TAStates
            dtype                       = dt
            Device                      = dvc
            ClausesPerClass             = cfg.ClausesPerClass
            InputSize                   = cfg.InputSize
            Classes                     = cfg.Classes
        }

    let dispose (tm:TM) =
        tm.Weights.Dispose()
        tm.Clauses.Dispose()
        tm.TMState.HighState.Dispose()
        tm.TMState.LowState.Dispose()
        tm.TMState.MidState.Dispose()
        tm.TMState.PayoutMatrix.Dispose()
        tm.TMState.PlrtySignClass.Dispose()
        tm.TMState.PolarityIndex.Dispose()
        tm.TMState.PolaritySign.Dispose()
        tm.TMState.Ones.Dispose()
        tm.TMState.Zeros.Dispose()
        tm.TMState.T2.Dispose()
        tm.TMState.TMinus.Dispose()
        tm.TMState.TPlus.Dispose()
        tm.TMState.Y0.Dispose()
        tm.TMState.Y1.Dispose()       

    let exportLearned (tm:TM) = 
        use clauses = tm.Clauses.to_type(torch.int32).cpu()
        use weights = tm.Weights.to_type(torch.int32).cpu()
        let taStates = clauses.data<int32>().ToArray()
        let clsWts   = weights.data<int32>().ToArray()
        taStates,clsWts

    let save (file:string) (tm:TM) =
        let cfg = tm.TMState.Config
        let taStates,clsWts = exportLearned tm
        let package = toState cfg,taStates,clsWts
        let ser = FsPickler.CreateXmlSerializer()
        use str = System.IO.File.Create file
        ser.Serialize(str,package)

    let load (device:torch.Device) (file:string) = 
        let ser = FsPickler.CreateXmlSerializer()
        let str = System.IO.File.OpenRead file
        let st,taStates,weights = ser.Deserialize<State*int32[]*int32[]>(str)
        let cfg = toConfig device st
        let tm = create cfg
        use initClauses = tm.Clauses
        use initWeights = tm.Weights
        let clauses = torch.tensor(taStates,dtype=cfg.dtype,device=cfg.Device,dimensions=initClauses.shape)
        let weights = torch.tensor(weights,dtype=cfg.dtype,device=cfg.Device,dimensions=initWeights.shape)
        let chunkedClauses = clauses.chunk(int64 cfg.Classes)
        let chunkedWts = weights.chunk(int64 cfg.Classes)
        let tmst = 
            {tm.TMState with
                ChunkedClauses = chunkedClauses
                ChunkedWeights = chunkedWts
            }
        {tm with 
            Clauses = clauses
            Weights = weights
            TMState = tmst
        }
