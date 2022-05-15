namespace FsTsetlin 
open TorchSharp
open System

type Config = 
    {
        s           : float32
        T           : float32
        TAStates    : int
        dtype       : torch.ScalarType
        Device      : torch.Device
        Clauses     : int
        InputSize   : int
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
        MinusOnes       : torch.Tensor
        TPlus           : torch.Tensor
        TMinus          : torch.Tensor
        T2              : torch.Tensor
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
        let ts = t.data<'a>().ToArray()
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
            use alExs = filter.any(1L,keepDim=true)                                         //true if all actions are 'exclude' for a clause
            if alExs.any().ToBoolean() then
                let adjOutput = torch.where(alExs,invrts.Zeros,taOutput)
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
        withPlry.sum().to_type(torch.float32)

module Train = 
    ///obtain +/- reward probabilities for each TA (for type I and II feedback)
    let rewardProb invrts (clauses:torch.Tensor) (clauseEvals:torch.Tensor) (X:torch.Tensor,y:torch.Tensor) =
        use filter = clauses.greater(invrts.MidState)
        use ce_t = clauseEvals.reshape(-1L,1L)
        (*polarity    literal     action  Cw  y   ->  p_reward *)
        use literal_f = X.expand_as(clauses).reshape([|1L;-1L|]).to_type(torch.int64)
        use action_f = torch.where(filter,invrts.Ones,invrts.Zeros).reshape([|1L;-1L|]).to_type(torch.int64)
        use cw_f = torch.hstack(ResizeArray[for _ in 1 .. int X.shape.[0] -> ce_t]).reshape([|1L;-1L|]).to_type(torch.int64)
        use y_f = y.expand_as(clauses).reshape([|1L;-1L|]).to_type(torch.int64)
        invrts.PayoutMatrix.index([|invrts.PolarityIndex; literal_f; action_f; cw_f; y_f|])

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
        use zeros = invrts.Zeros.reshape(pReward.shape)
        use ones = invrts.Ones.reshape(pReward.shape)
        use minusOnes = invrts.MinusOnes.reshape(pReward.shape)
        use negFeedback = minusOnes.where(negRwdFltr,zeros)
        use posFeedback = ones.where(posRwdFltr,zeros)
        negFeedback + posFeedback                                   //final feedback tensor with -1, +1 or 0 values 

    ///calculate feedback incr/decr values based on selected feedback reward(+)/penalty(-)/ignore(0) and TA state
    let feedbackIncrDecr invrts (clauses:torch.Tensor) (feedback:torch.Tensor) =
        use filterGreater = clauses.greater(invrts.MidState)
        use filterLessEqual = clauses.less_equal(invrts.MidState)
        use highs = torch.where(filterGreater,invrts.Ones,invrts.Zeros)
        use lows = torch.where(filterLessEqual,invrts.MinusOnes,invrts.Zeros)
        use feedbackPolarity = highs + lows
        feedbackPolarity.mul(feedback.reshape(clauses.shape))

    ///inplace update clauses
    let updateClauses_ invrts (clauses:torch.Tensor) (incrDecr:torch.Tensor) =
        clauses.add_(incrDecr).clamp_(invrts.LowState,invrts.HighState)

    ///udpated clauses as a new tensor (for debugging purposes)
    let updateClauses invrts (clauses:torch.Tensor) (incrDecr:torch.Tensor) =
        let clss = clauses.add(incrDecr)
        clss.clamp_(invrts.LowState,invrts.HighState)

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
        let rng = System.Random()
        let numTAs = cfg.Clauses * (cfg.InputSize * 2)
        let initialState = [|for i in 0..numTAs-1 -> if rng.NextDouble() < 0.5 then cfg.TAStates else cfg.TAStates+1|]
        let plrtyBin = [|for i in 0 .. cfg.Clauses-1 -> i % 2|]
        let plrtySgn = [|for i in 0 .. cfg.Clauses-1 -> if i%2 = 0 then -1 else 1|]           
        let clauses = torch.tensor(initialState, dtype=cfg.dtype, dimensions = [|int64 cfg.Clauses; cfg.InputSize * 2 |> int64|], device=cfg.Device)
        let polarity = torch.tensor(plrtyBin, dtype=cfg.dtype, device=cfg.Device, dimensions=[|clauses.shape.[0]; 1L|])
        let polarityIdx = polarity.expand_as(clauses).reshape([|1L;-1L|]).to_type(torch.int64)
        let tPlus = torch.tensor(cfg.T, device=cfg.Device)
        let tMinus = torch.tensor(-cfg.T, device=cfg.Device)
        let t2 = torch.tensor(2.0f * cfg.T,device=cfg.Device)

        let cache =
            {
                PolarityIndex   = polarityIdx
                PolaritySign    = torch.tensor(plrtySgn, dtype=cfg.dtype, device=cfg.Device)
                PayoutMatrix    = torch.tensor(payout cfg.s, dimensions = [|2L;2L;2L;2L;2L|], device=cfg.Device)   
                MidState        = torch.tensor([|cfg.TAStates|],dtype=cfg.dtype, device=cfg.Device)
                LowState        = torch.tensor([|1|],dtype=cfg.dtype,device=cfg.Device)
                HighState       = torch.tensor([|2*cfg.TAStates|],dtype=cfg.dtype,device=cfg.Device)
                Zeros           = torch.zeros_like(clauses)
                Ones            = torch.ones_like(clauses)
                MinusOnes       = torch.full_like(clauses,-1)
                Config          = cfg
                TPlus           = tPlus
                TMinus          = tMinus
                T2              = t2
            }
        {
            Clauses     = clauses
            Invariates  = cache
        }

    let train (X,y) (tm:TM) =
        Train.trainStep tm.Invariates tm.Clauses (X,y)

    let trainBatch (X:torch.Tensor,y:torch.Tensor) (tm:TM) =
        let batchSze = X.shape.[0]
        for i in 0L .. batchSze - 1L do
            train (X.[i],y.[i]) tm

    let eval X (tm:TM) =
        use taEvals = Eval.evalTA tm.Invariates false tm.Clauses X //num_clauses * input
        use clauseEvals = Eval.andClause tm.Invariates taEvals
        use v = Eval.sumClauses tm.Invariates clauseEvals
        if v.ToSingle() > 0.f then 1 else 0

    let evalBatch (X:torch.Tensor) (tm:TM) =
        let batchSze = X.shape.[0]
        [|for i in 0L .. batchSze - 1L do
            eval X.[i] tm
        |]

            
            
