namespace FsTsetlin 
open TorchSharp
open System

type Config = 
    {
        s                   : float32
        T                   : float32
        TAStates            : int
        dtype               : torch.ScalarType
        Device              : torch.Device
        ClausesPerClass     : int
        InputSize           : int
        Classes             : int // 2 or more
    }

/// cache invariates for optimization
type Invariates = 
    {
        Config          : Config
        PolarityIndex   : torch.Tensor
        PolaritySign    : torch.Tensor
        PayoutMatrix    : torch.Tensor
        MidState        : torch.Tensor
        LowState        : torch.Tensor
        HighState       : torch.Tensor
        Ones            : torch.Tensor
        Zeros           : torch.Tensor
        ClassOnes       : torch.Tensor
        ClassZeros      : torch.Tensor
        ClassMinusOnes  : torch.Tensor
        TPlus           : torch.Tensor
        TMinus          : torch.Tensor
        T2              : torch.Tensor
        Y1              : torch.Tensor
        Y0              : torch.Tensor
        IsBinary        : bool
    }

type TM =
    {
        Clauses         : torch.Tensor
        Invariates      : Invariates
    }

module Utils =
    type D<'a> = F of 'a[] | T of D<'a>[]

    //utility function to get raw tensor data as a recursive structure for debugging purposes
    let tensorData<'a when 'a: (new: unit -> 'a) and  'a: struct and 'a :> ValueType>(t:torch.Tensor) = 
        let ts = if t.device<>torch.CPU then t.cpu().data<'a>().ToArray() else t.data<'a>().ToArray()
        let rdims =
            t.shape 
            |> Array.map int 
            |> Array.rev            //start with inner most dimension
            |> Array.toList
        let rec loop ds (xs:D<'a>) =
            match ds,xs with
            | [],_                        -> xs
            | d::[],T ds when d=ds.Length -> T ds
            | d::[],F ds when d=ds.Length -> F ds
            | d::rest,T ds -> loop rest (ds |> Array.chunkBySize d |> Array.map T |> T)
            | d::rest,F ds -> loop rest (ds |> Array.chunkBySize d |> Array.map F |> T)
        loop rdims (F ts)
        
module Eval = 
    
    ///eval each literal against each tsetlin automaton (TA) in the clauses
    let evalTA invrts trainMode (clauses:torch.Tensor)  (input:torch.Tensor) = 
        use input2   = input.broadcast_to(clauses.shape)                                    //make input the same shape as clauses
        use filter   = clauses.greater(invrts.MidState)                                     //determine 'include'/'exclude' actions
        let taOutput = torch.where(filter,input2,invrts.Ones)                               //take input if action = include else 1 (which skips excluded input when and'ing)
        if not trainMode then   
            use oneInc = filter.any(1L,keepDim=true)                                        //true if there is at least 1 'include' per clause
            if oneInc.any().ToBoolean() then
                let adjOutput = torch.where(oneInc,taOutput,invrts.Zeros)
                taOutput.Dispose()
                adjOutput
            else
                taOutput
        else
            taOutput

    ///AND the outputs of TAs by clause
    let andClause invrts (evals:torch.Tensor)  =
        use prods = evals.cumprod(1L,``type``=invrts.Config.dtype)
        prods.[torch.TensorIndex.Ellipsis,torch.TensorIndex.Single(-1L)]

    ///sum positive and negative polarity clause outputs
    let sumClauses invrts (clauseEvals:torch.Tensor) =
        use withPlry = clauseEvals.mul(invrts.PolaritySign)
        withPlry.sum(``type``=torch.float32)

    ///sum positive and negative polarity clause outputs per class 
    let sumClausesMulticlass invrts (clauseEvals:torch.Tensor) =
        use withPlry = clauseEvals.mul(invrts.PolaritySign)
        use byClass = withPlry.reshape( int64 invrts.Config.Classes, -1L )
        byClass.sum(1L, ``type``= torch.float32)

module Train = 
    ///obtain +/- reward probabilities for each TA (for type I and II feedback)
    let rewardProb invrts (clauses:torch.Tensor) (clauseEvals:torch.Tensor) (X:torch.Tensor,y:torch.Tensor) =
        use filter = clauses.greater(invrts.MidState)
        use ce_t = clauseEvals.reshape(-1L,1L)
        (*polarity    literal     action  Cw  y   ->  p_reward *)
        use literal_f = X.expand_as(clauses).reshape([|1L;-1L|]).to_type(torch.int64)
        use action_f = torch.where(filter,invrts.ClassOnes,invrts.ClassZeros).reshape([|1L;-1L|]).to_type(torch.int64)
        use cw_f = torch.hstack(ResizeArray[for _ in 1 .. int X.shape.[0] -> ce_t]).reshape([|1L;-1L|]).to_type(torch.int64)
        use y_f = y.expand_as(clauses).reshape([|1L;-1L|]).to_type(torch.int64)
        use plrt_f = invrts.PolarityIndex.reshape([|1L;-1L|])
        invrts.PayoutMatrix.index([|plrt_f; literal_f; action_f; cw_f; y_f|])

    ///postive, negative or no feedback for each TA
    let taFeeback invrts (v:torch.Tensor) (pReward:torch.Tensor) (y:torch.Tensor) =
        use selY1 = (invrts.TPlus - v.clamp(invrts.TMinus,invrts.TPlus)) / invrts.T2  //feedback selection prob. when y=1
        use selY0 = (invrts.TPlus + v.clamp(invrts.TMinus,invrts.TPlus)) / invrts.T2  //feedback selection prob. when y=0 
        use uRand = torch.rand_like(pReward)              //tensor of uniform random values for Feedback selection
        use feebackFilter =                               //bool tensor; true if random <= [selY0 or selY1] (based on the value of y )
            if y.ToSingle() <= 0.f then 
                uRand.less_equal(selY0) 
            else 
                uRand.less_equal(selY1)      
        use zeros = torch.zeros_like(pReward)                   //zero filled tensor - same shape as reward probabilities
        use pRewardSel = pReward.where(feebackFilter,zeros)     //keep reward prob if filter tensor value is true, zero otherwise
        use negRewards = pRewardSel.minimum(zeros)          //separate out negative reward prob.
        use posRewards = pRewardSel.maximum(zeros)          //separate out postive reward prob.
        use uRandRwrd = torch.rand_like(pRewardSel)         //tensor of uniform random values for Reward/Penalty selection
        use negRwdFltr = uRandRwrd.less_equal(negRewards.abs_()) //negative reward filter reflecting random selection
        use posRwdFltr = uRandRwrd.less_equal(posRewards)        //positive reward filter reflecting random selection
        use zeros = invrts.ClassZeros.reshape(pReward.shape)
        use ones = invrts.ClassOnes.reshape(pReward.shape)
        use minusOnes = invrts.ClassMinusOnes.reshape(pReward.shape)
        use negFeedback = minusOnes.where(negRwdFltr,zeros)
        use posFeedback = ones.where(posRwdFltr,zeros)
        let fb = negFeedback + posFeedback                      //final feedback tensor with -1, +1 or 0 values 
        fb

    ///calculate feedback incr/decr values based on selected feedback reward(+)/penalty(-)/ignore(0) and TA state
    let feedbackIncrDecr invrts (clauses:torch.Tensor) (feedback:torch.Tensor) =
        use filterGreater = clauses.greater(invrts.MidState)
        use filterLessEqual = clauses.less_equal(invrts.MidState)
        use highs = torch.where(filterGreater,invrts.ClassOnes,invrts.ClassZeros)
        use lows = torch.where(filterLessEqual,invrts.ClassMinusOnes,invrts.ClassZeros)
        use feedbackPolarity = highs + lows
        feedbackPolarity.mul(feedback.reshape(clauses.shape))

    ///inplace update clauses
    let updateClauses_ invrts (clauses:torch.Tensor) (incrDecr:torch.Tensor) =
        clauses.add_(incrDecr).clamp_(invrts.LowState,invrts.HighState)

    ///udpated clauses as a new tensor (for debugging purposes)
    let updateClauses invrts (clauses:torch.Tensor) (incrDecr:torch.Tensor) =
        let clss = clauses.add(incrDecr)
        clss.clamp_(invrts.LowState,invrts.HighState)

    ///update the clauses for a single class - used in multi-class scenario
    let updateClass invrts clauses clauseEvals v (X,y) = 
        use pReward = rewardProb invrts clauses clauseEvals (X,y)
        use feedback = taFeeback invrts v pReward y
        use fbIncrDecr = feedbackIncrDecr invrts clauses feedback 
        updateClauses_ invrts clauses fbIncrDecr |> ignore

    ///update clauses on single input - optimized for producton
    let trainStepMulticlass invrts clauses (X,y:torch.Tensor) = 
        use taEvals = Eval.evalTA invrts true clauses X //num_clauses * input
        use clauseEvals = Eval.andClause invrts taEvals
        use byClassSum = Eval.sumClausesMulticlass invrts clauseEvals
        let idx1 = y.ToInt64()
        let remClss = [|for i in 0L .. int64 (invrts.Config.Classes - 1) do if i <> idx1 then yield i |] 
        use tRemClss = torch.tensor(remClss,dtype=torch.int64,device=invrts.Config.Device)
        use notYIdx = torch.randint(int tRemClss.shape.[0],[|1|],dtype=torch.int64,device=invrts.Config.Device)
        let notY = tRemClss.[notYIdx]
        let chunkedClauses = clauses.chunk(int64 invrts.Config.Classes)
        let chunkedEvals   = clauseEvals.chunk(int64 invrts.Config.Classes)
        //when class = y
        use v1Clauses = chunkedClauses.[y.ToInt32()]
        use v1Evals   = chunkedEvals.[y.ToInt32()]
        let v1        = byClassSum.[y]
        updateClass invrts v1Clauses v1Evals v1 (X,invrts.Y1)
        //when class not y (randomly chosen)
        use v0Clauses = chunkedClauses.[notY.ToInt32()]
        use v0Evals   = chunkedEvals.[notY.ToInt32()]
        let v0        = byClassSum.[notY]
        updateClass invrts v0Clauses v0Evals v0 (X,invrts.Y0)        
        //dispose chunked views
        chunkedClauses |> Array.iter (fun x -> x.Dispose())
        chunkedEvals   |> Array.iter (fun x -> x.Dispose())

    ///update clauses on single input - optimized for producton
    let trainStep invrts clauses (X,y) = 
        use taEvals = Eval.evalTA invrts true clauses X //num_clauses * input
        use clauseEvals = Eval.andClause invrts taEvals
        use v = Eval.sumClauses invrts clauseEvals
        use pReward = rewardProb invrts clauses clauseEvals (X,y)
        use feedback = taFeeback invrts v pReward y
        use fbIncrDecr = feedbackIncrDecr invrts clauses feedback 
        updateClauses_ invrts clauses fbIncrDecr |> ignore

    //debug version of update that returns intermediate results
    let trainStepDbg invrts clauses (X,y)  =
        let taEvals = Eval.evalTA invrts true clauses X //num_clauses * input
        let clauseEvals = Eval.andClause invrts taEvals
        let v = Eval.sumClauses invrts clauseEvals
        let pReward = rewardProb invrts clauses clauseEvals (X,y)
        let feedback = taFeeback invrts v pReward y
        let fbIncrDecr = feedbackIncrDecr invrts clauses feedback 
        let updtClss = updateClauses invrts clauses fbIncrDecr 
        taEvals,clauseEvals,v,pReward,feedback,fbIncrDecr,updtClss

module TM =
    let inaction = 0.0f
    let pnlty    = -1.0f

    let payout s = 
        let ``1/s``     = 1.0f / s
        let ``(s-1)/s`` = (s - 1.0f) / s
        [|
        (*polarity    literal     action  Cw  y     p_reward *)
        (*0           0           0       0   0 *)  ``1/s``         //0
        (*0           0           0       0   1 *)  inaction        //1
        (*0           0           0       1   0 *)  ``1/s``         //2
        (*0           0           0       1   1 *)  pnlty           //3
        (*0           0           1       0   0 *)  -``1/s``        //4
        (*0           0           1       0   1 *)  inaction        //5
        (*0           0           1       1   0 *)  inaction        //6
        (*0           0           1       1   1 *)  inaction        //7
        (*0           1           0       0   0 *)  ``1/s``         //8
        (*0           1           0       0   1 *)  inaction        //9
        (*0           1           0       1   0 *)  -``(s-1)/s``    //10
        (*0           1           0       1   1 *)  inaction        //11
        (*0           1           1       0   0 *)  -``1/s``        //12
        (*0           1           1       0   1 *)  inaction        //13
        (*0           1           1       1   0 *)  ``(s-1)/s``     //14
        (*0           1           1       1   1 *)  inaction        //15
        (*1           0           0       0   0 *)  inaction        //16
        (*1           0           0       0   1 *)  ``1/s``         //17
        (*1           0           0       1   0 *)  pnlty           //18
        (*1           0           0       1   1 *)  ``1/s``         //19
        (*1           0           1       0   0 *)  inaction        //20
        (*1           0           1       0   1 *)  -``1/s``        //21
        (*1           0           1       1   0 *)  inaction        //22
        (*1           0           1       1   1 *)  inaction        //23
        (*1           1           0       0   0 *)  inaction        //24
        (*1           1           0       0   1 *)  ``1/s``         //25
        (*1           1           0       1   0 *)  inaction        //26
        (*1           1           0       1   1 *)  -``(s-1)/s``    //27
        (*1           1           1       0   0 *)  inaction        //28
        (*1           1           1       0   1 *)  -``1/s``        //29
        (*1           1           1       1   0 *)  inaction        //30
        (*1           1           1       1   1 *)  ``(s-1)/s``     //31
        |]

    let create (cfg:Config) =
        if cfg.ClausesPerClass % 2 <> 0 then failwithf "clauses per class should be an even number (for negative/positive polarity)"
        let rng = System.Random()
        let isBinary = cfg.Classes = 2
        let numTAs = cfg.Classes * cfg.ClausesPerClass * (cfg.InputSize * 2)
        let initialState = [|for i in 1..numTAs -> if rng.NextDouble() < 0.5 then cfg.TAStates else cfg.TAStates+1|]
        let clauses = torch.tensor(initialState, dtype=cfg.dtype, dimensions = [|int64 (cfg.Classes * cfg.ClausesPerClass); int64 (cfg.InputSize * 2) |], device=cfg.Device)
        let plrtySgn = [|for i in 1 .. (cfg.Classes * cfg.ClausesPerClass) -> if i % 2 = 0 then -1 else 1|]  //polarity sign +1/-1; used for summing clause evaluation
        let numClauses = if isBinary then cfg.Classes * cfg.ClausesPerClass else cfg.ClausesPerClass
        let plrtyBin = [|for i in 1 .. numClauses -> i % 2|]                                                  //polarity index 1/0; used for indexing into the payout matrix
        let polarityIdx = 
            if isBinary then
                let polarity = torch.tensor(plrtyBin, dtype=torch.int64, device=cfg.Device, dimensions=[|clauses.shape.[0]; 1L|])
                polarity.expand_as(clauses).reshape([|1L;-1L|])
            else
                let polarity = torch.tensor(plrtyBin, dtype=torch.int64, device=cfg.Device, dimensions=[|int64 cfg.ClausesPerClass; 1L|])
                polarity.broadcast_to(int64 cfg.ClausesPerClass, int64 (2 * cfg.InputSize))
        let tPlus = torch.tensor(cfg.T, device=cfg.Device)
        let tMinus = torch.tensor(-cfg.T, device=cfg.Device)
        let t2 = torch.tensor(2.0f * cfg.T,device=cfg.Device)
        let ones  = torch.ones_like(clauses)
        let zeros = torch.zeros_like(clauses)
        let cache =
            {
                PolarityIndex   = polarityIdx
                PolaritySign    = torch.tensor(plrtySgn, dtype=cfg.dtype, device=cfg.Device)
                PayoutMatrix    = torch.tensor(payout cfg.s, dimensions = [|2L;2L;2L;2L;2L|], device=cfg.Device)   
                MidState        = torch.tensor([|cfg.TAStates|],dtype=cfg.dtype, device=cfg.Device)
                LowState        = torch.tensor([|1|],dtype=cfg.dtype,device=cfg.Device)
                HighState       = torch.tensor([|2*cfg.TAStates|],dtype=cfg.dtype,device=cfg.Device)
                Zeros           = zeros
                Ones            = ones
                ClassOnes       = if isBinary then ones else torch.full_like(polarityIdx, 1,dtype=cfg.dtype)
                ClassMinusOnes  = if isBinary then torch.full_like(ones,-1) else torch.full_like(polarityIdx,-1,dtype=cfg.dtype)
                ClassZeros      = if isBinary then zeros else torch.full_like(polarityIdx, 0,dtype=cfg.dtype)
                Config          = cfg
                TPlus           = tPlus
                TMinus          = tMinus
                T2              = t2
                Y1              = 1L.ToTensor(device=cfg.Device)
                Y0              = 0L.ToTensor(device=cfg.Device)
                IsBinary        = isBinary
            }
        {
            Clauses     = clauses
            Invariates  = cache
        }

    let train (X,y) (tm:TM) =
        if tm.Invariates.IsBinary then
            Train.trainStep tm.Invariates tm.Clauses (X,y)
        else
            Train.trainStepMulticlass tm.Invariates tm.Clauses (X,y)

    let trainBatch (X:torch.Tensor,y:torch.Tensor) (tm:TM) =
        let batchSze = X.shape.[0]
        for i in 0L .. batchSze - 1L do
            train (X.[i],y.[i]) tm

    let predict X (tm:TM) =
        use taEvals = Eval.evalTA tm.Invariates false tm.Clauses X //num_clauses * input
        use clauseEvals = Eval.andClause tm.Invariates taEvals
        if tm.Invariates.Config.Classes > 2 then
            use byClassSum = Eval.sumClausesMulticlass tm.Invariates clauseEvals
            use idx = byClassSum.argmax()
            idx.ToInt32()
        else
            use v = Eval.sumClauses tm.Invariates clauseEvals
            if v.ToSingle() > 0.f then 1 else 0

    let predictBatch (X:torch.Tensor) (tm:TM) =
        let batchSze = X.shape.[0]
        [|for i in 0L .. batchSze - 1L do
            predict X.[i] tm
        |]

            
            
