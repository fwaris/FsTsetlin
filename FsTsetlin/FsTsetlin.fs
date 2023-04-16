namespace FsTsetlin 
open TorchSharp
open System
open MBrace.FsPickler

type Config = 
    {
        s                           : float32
        T                           : float32
        TAStates                    : int
        Device                      : torch.Device
        ClausesPerClass             : int
        InputSize                   : int
        Classes                     : int // 2 or more
        BoostTruePositiveFeedback   : bool
        MaxWeight                   : int
    }
    with 
        static member Default  = 
            {
                s                           = 3.9f
                T                           = 15.0f
                TAStates                    = 100
                Device                      = torch.CPU
                ClausesPerClass             = 100
                InputSize                   = 0
                Classes                     = 2
                BoostTruePositiveFeedback   = false
                MaxWeight                   = 1
            }

/// cached internal state
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
        MinusOnes       : torch.Tensor
        TPlus           : torch.Tensor
        TMinus          : torch.Tensor
        T2              : torch.Tensor
        Y1              : torch.Tensor
        Y0              : torch.Tensor
        IsBinary        : bool
        ChunkedClauses  : torch.Tensor[]
        ChunkedWeights  : torch.Tensor[]
        MaxWeightT      : torch.Tensor
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
                use oneInc = actions.any(1L,keepdim=true)               //true if there is at least 1 'include' per clause
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
        use fltr2 = fltr.all(dim=1L)
        torch.where(fltr2,tmst.Ones,tmst.Zeros)

    ///sum positive and negative polarity clause outputs for a single class
    let sumClass (tmst:TMState) (clauseEvals:torch.Tensor) (weights:torch.Tensor) =
        if tmst.Config.MaxWeight > 1 then
            use wtd = clauseEvals.mul_(weights)
            use withPlry = wtd.mul_(tmst.PlrtySignClass)
            withPlry.sum(``type``=torch.float32)
        else
            use withPlry = clauseEvals.mul_(tmst.PlrtySignClass)
            withPlry.sum(``type``=torch.float32)


    ///sum positive and negative polarity clause outputs per class for all classes
    let sumClausesMulticlass tmst (clauseEvals:torch.Tensor) (weights:torch.Tensor) =
        if tmst.Config.MaxWeight > 1 then            
            use wts = weights.reshape(1L,-1L)
            use wtd = clauseEvals.mul_(wts)
            use withPlry = wtd.mul_(tmst.PolaritySign)
            use byClass = withPlry.reshape( int64 tmst.Config.Classes, -1L )
            byClass.sum(1L, ``type``= torch.float32)
        else
            use withPlry = clauseEvals.mul_(tmst.PolaritySign)
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
    let taFeeback tmst (feedbackFilter:torch.Tensor) (pReward:torch.Tensor) =
        use pr1 = pReward.reshape(feedbackFilter.shape.[0], -1L)
        use ff1 =  feedbackFilter.reshape(-1L,1L)
        use ff2 = ff1.expand_as(pr1)  
        use feedbackExpanded = ff2.reshape(1L,-1L)
        use uRandRwrd = torch.rand_like(pReward)         //tensor of uniform random values for Reward/Penalty selection
        use pRwrd_abs = pReward.abs()
        use pSelFltr = uRandRwrd.less_equal(pRwrd_abs)
        use pFilter = feedbackExpanded.logical_and(pSelFltr)
        use rwdSign = pReward.sign()
        use rwdSignInt = rwdSign.to_type(torch.int8)
        let fb = rwdSignInt.where(pFilter,tmst.Zeros)
        fb

    ///calculate feedback incr/decr values based on selected feedback reward(+)/penalty(-)/ignore(0) and TA state
    let feedbackIncrDecr tmst (actions:torch.Tensor) (feedback:torch.Tensor) =
        use one = torch.tensor([|1|],dtype=torch.int16, device=tmst.Config.Device)
        use minusOne = torch.tensor([|-1|],dtype=torch.int16, device=tmst.Config.Device)
        use filterBin = torch.where(actions,one,minusOne)
        let fp = filterBin.mul(feedback.reshape(actions.shape))
        fp

    ///update the clauses and weights for a single class
    let updateClass tmst actions (clauses:torch.Tensor) clauseEvals (weights:torch.Tensor) (v:torch.Tensor) (X,y:torch.Tensor) = 
        use uRand = torch.rand_like(clauseEvals,dtype=torch.ScalarType.Float16)       //tensor of uniform random values for Feedback selection
        let yVal = y.ToSingle()
        let vVal = v.ToSingle()
        use feebackFilter =                                                           //bool tensor; true if random <= [selY0 or selY1] (based on the value of y )
            if yVal <= 0.f then 
                use selY0 = (tmst.TPlus + v.clamp(tmst.TMinus,tmst.TPlus)) / tmst.T2  //feedback selection prob. when y=0 
                uRand.less_equal(selY0) 
            else 
                use selY1 = (tmst.TPlus - v.clamp(tmst.TMinus,tmst.TPlus)) / tmst.T2  //feedback selection prob. when y=1
                uRand.less_equal(selY1)              
        use pReward = rewardProb tmst actions clauseEvals (X,y)
        use feedback = taFeeback tmst feebackFilter pReward
        use fbIncrDecr = feedbackIncrDecr tmst actions feedback 
        clauses.add_(fbIncrDecr).clamp_(tmst.LowState,tmst.HighState) |> ignore
        if tmst.Config.MaxWeight > 1 then
            use oneClauses = clauseEvals.greater(tmst.Zeros)                //clauses that eval to 1
            use oneClausesFb = oneClauses.logical_and(feebackFilter)        //1-clauses filtered by feedback
            let wtsDelta =
                if vVal > 0.f && yVal > 0.f then
                    tmst.Ones
                else
                    tmst.MinusOnes
            use weightChanges = torch.where(oneClausesFb,wtsDelta,tmst.Zeros)
            use withPolarity = weightChanges.mul(tmst.PlrtySignClass)
            weights.add_(withPolarity).clamp_(tmst.Ones,tmst.MaxWeightT) |> ignore

    let trainStepMulticlass tmst (clauses:torch.Tensor) (X,y:torch.Tensor) = 
        let idx1 = y.ToInt64()
        use notY = 
            if tmst.IsBinary then
                let notYIdx = if idx1 = 1L then 0L else 1L            
                torch.tensor([|notYIdx|],dtype=torch.int64,device=tmst.Config.Device)
            else
                let remClss = [|for i in 0L .. int64 (tmst.Config.Classes - 1) do if i <> idx1 then yield i |] 
                use tRemClss = torch.tensor(remClss,dtype=torch.int64,device=tmst.Config.Device)
                use notYIdx = torch.randint(int tRemClss.shape.[0],[|1|],dtype=torch.int64,device=tmst.Config.Device) //all randomness should be from torch
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
        updateClass tmst v1Actions v1Clauses v1ClsEvals w1 v1 (X,tmst.Y1)
        //when class not y (randomly chosen when classes > 2)
        let notYIdx = notY.ToInt32()
        let v0Clauses = tmst.ChunkedClauses.[notYIdx]
        let v0Actions,v0TAEvals = Eval.evalTA tmst true v0Clauses X
        let v0Actions = v0Actions
        let v0TAEvals = v0TAEvals
        use v0ClsEvals = Eval.andClause tmst v0TAEvals
        let w0        = tmst.ChunkedWeights.[notYIdx]
        use v0        = Eval.sumClass tmst v0ClsEvals w0
        updateClass tmst v0Actions v0Clauses v0ClsEvals w0 v0 (X,tmst.Y0)        

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
        use _g  = torch.no_grad()       
        if cfg.ClausesPerClass % 2 <> 0 then failwithf "clauses per class should be an even number (for negative/positive polarity)"
        let rng = System.Random()
        let isBinary = cfg.Classes = 2
        let numTAs = cfg.Classes * cfg.ClausesPerClass * (cfg.InputSize * 2)
        let initialState = [|for i in 1..numTAs -> if rng.NextDouble() < 0.5 then 0 else 1|]
        let clauses = torch.tensor(initialState, dtype=torch.int16, dimensions = [|int64 (cfg.Classes * cfg.ClausesPerClass); int64 (cfg.InputSize * 2) |], device=cfg.Device)
        

        let plrtySgnT =                                                                 
            let plrtySgn = [|1; -1|]                                                        //use broadcasting to reduce memory 
            use plrtySgn_1 = torch.tensor(plrtySgn,dtype=torch.int8,device=cfg.Device)
            use plrtySgn_2 = plrtySgn_1.reshape(1L,2L)
            use plrtySgn_3 = plrtySgn_2.broadcast_to(int64 (cfg.Classes * cfg.ClausesPerClass / 2), 2L)
            plrtySgn_3.reshape(int64 (cfg.Classes * cfg.ClausesPerClass))

        let plrtySgnClsT =                                                                 
            let plrtySgn = [|1; -1|]                                                        //use broadcasting to reduce memory 
            use plrtySgn_1 = torch.tensor(plrtySgn,dtype=torch.int8,device=cfg.Device)
            use plrtySgn_2 = plrtySgn_1.reshape(1L,2L)
            use plrtySgn_3 = plrtySgn_2.broadcast_to(int64 (cfg.ClausesPerClass / 2), 2L)
            plrtySgn_3.reshape(int64 cfg.ClausesPerClass)

        let polarityIdx = 
            let plrtyBin = [|for i in 1 .. cfg.ClausesPerClass -> i % 2|]                                                  //polarity index 1/0; used for indexing into the payout matrix
            let polarity = torch.tensor(plrtyBin, dtype=torch.int64, device=cfg.Device, dimensions=[|int64 cfg.ClausesPerClass; 1L|])
            polarity.broadcast_to(int64 cfg.ClausesPerClass, int64 (2 * cfg.InputSize))

        let tPlus = torch.tensor(cfg.T, device=cfg.Device)
        let tMinus = torch.tensor(-cfg.T, device=cfg.Device)
        let t2 = torch.tensor(2.0f * cfg.T,device=cfg.Device)
        let ones  = torch.tensor([|1|], dtype = torch.int8, device = cfg.Device)
        let zeros = torch.tensor([|0|], dtype = torch.int8, device = cfg.Device)
        let minusOnes = torch.tensor([|-1|], dtype = torch.int8, device = cfg.Device)
        let maxW = torch.tensor([|cfg.MaxWeight|], dtype = torch.int8, device = cfg.Device) 
        let payout = payout cfg.BoostTruePositiveFeedback cfg.s
        let weights = torch.tensor([|for i in 1 .. cfg.Classes*cfg.ClausesPerClass -> 1|], dtype = torch.int8, device = cfg.Device, dimensions=[| int64 cfg.Classes; int64 cfg.ClausesPerClass |])
        let chunkedClauses = clauses.chunk(int64 cfg.Classes)
        let chunkedWts = weights.chunk(int64 cfg.Classes)
        let tmState =
            {
                PolarityIndex   = polarityIdx
                PolaritySign    = plrtySgnT
                PlrtySignClass  = plrtySgnClsT
                PayoutMatrix    = torch.tensor(payout, dimensions = [|2L;2L;2L;2L;2L|], device=cfg.Device)   
                MidState        = torch.tensor([|0|],dtype=torch.int16, device=cfg.Device)
                LowState        = torch.tensor([|-cfg.TAStates|],dtype=torch.int16,device=cfg.Device)
                HighState       = torch.tensor([|cfg.TAStates|],dtype=torch.int16,device=cfg.Device)
                Zeros           = zeros
                Ones            = ones
                MinusOnes       = minusOnes
                Config          = cfg
                TPlus           = tPlus
                TMinus          = tMinus
                T2              = t2
                Y1              = 1L.ToTensor(device=cfg.Device)
                Y0              = 0L.ToTensor(device=cfg.Device)
                IsBinary        = isBinary
                ChunkedClauses  = chunkedClauses
                ChunkedWeights  = chunkedWts
                MaxWeightT      = maxW
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
            System.GC.Collect(0,GCCollectionMode.Forced)

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
        let rs = 
            [|for i in 0L .. batchSze - 1L do
                predict X.[i] tm
            |]
        System.GC.Collect(0,GCCollectionMode.Forced)
        rs

    type private State = 
        {
            s                           : float32
            T                           : float32
            TAStates                    : int
            ClausesPerClass             : int
            InputSize                   : int
            Classes                     : int // 2 or more            
            BoostTruePositiveFeedback   : bool
            MaxWeight                   : int
        }

    let private toState (cfg:Config) = 
        {
            s                           = cfg.s
            T                           = cfg.T
            BoostTruePositiveFeedback   = cfg.BoostTruePositiveFeedback
            TAStates                    = cfg.TAStates
            ClausesPerClass             = cfg.ClausesPerClass
            InputSize                   = cfg.InputSize
            Classes                     = cfg.Classes
            MaxWeight                   = cfg.MaxWeight
        }

    let private toConfig dvc (cfg:State) : Config = 
        {
            s                           = cfg.s
            T                           = cfg.T
            BoostTruePositiveFeedback   = cfg.BoostTruePositiveFeedback
            TAStates                    = cfg.TAStates
            Device                      = dvc
            ClausesPerClass             = cfg.ClausesPerClass
            InputSize                   = cfg.InputSize
            Classes                     = cfg.Classes
            MaxWeight                   = cfg.MaxWeight
        }

    /// release GPU or CPU memory associated with this TM
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

    ///export TAstates and weights as flat integer arrays
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
        let clauses = torch.tensor(taStates,dtype=torch.int16,device=cfg.Device,dimensions=initClauses.shape)
        let weights = torch.tensor(weights,dtype=torch.int16,device=cfg.Device,dimensions=initWeights.shape)
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
