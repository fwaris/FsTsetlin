namespace FsTsetlin 
open TorchSharp

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

type TM =
    {
        Clauses         : torch.Tensor
        PolarityIndex   : torch.Tensor
        PolaritySign    : torch.Tensor
        PayoutMatrix    : torch.Tensor
        Config          : Config
    }
        
module Eval = 
        
    //eval each literal against each tsetlin automaton (TA) in the clauses
    let evalTA cfg trainMode (clauses:torch.Tensor)  (input:torch.Tensor) = 
        let input2   = input.broadcast_to(clauses.shape)                     //make input the same shape as clauses
        use tMid     = torch.tensor(cfg.TAStates, device=cfg.Device)
        use filter   = clauses.greater(tMid)                                 //determine 'include'/'exclude' actions
        let ones     = torch.ones_like(input2)  
        let taOutput = torch.where(filter,input2,ones)                      //take input if action = include else 1 (which skips excluded input when and'ing)
        if not trainMode then
            use alExs = filter.any(1L,keepDim=true)                         //true if all actions are 'exclude' for a clause
            if alExs.any().ToBoolean() then
                let zeros = torch.zeros_like(input2)                        //at evaluation time, clauses with all actions excluded return 0
                let adjOutput = torch.where(alExs,zeros,taOutput)
                taOutput.Dispose()
                adjOutput
            else
                taOutput
        else
            taOutput
            

    //AND the outputs of TAs by clause
    let andClause cfg (evals:torch.Tensor)  =
        use prods = evals.cumprod(1L,``type``=cfg.dtype)
        prods.[torch.TensorIndex.Ellipsis,torch.TensorIndex.Single(-1L)]

    //sum positive and negative polarity clause outputs
    let sumClauses cfg (clauseEvals:torch.Tensor) (plrtySgn:torch.Tensor) =
        use withPlry = clauseEvals.mul(plrtySgn)
        withPlry.sum().to_type(torch.float32)


module Train = 
    //obtain +/- reward probabilities for each TA (for type I and II feedback)
    let rewardProb cfg 
        (payoutMatrix:torch.Tensor) (clauses:torch.Tensor) (clauseEvals:torch.Tensor) (polarityIndex:torch.Tensor) (X:torch.Tensor,y:torch.Tensor) =
        use tMid = torch.tensor([|cfg.TAStates|], device=cfg.Device)
        use filter = clauses.greater(tMid)
        use zs = torch.zeros_like(clauses, device=cfg.Device)
        use os = torch.ones_like(clauses, device=cfg.Device)
        let ce_t = clauseEvals.reshape(-1L,1L)
        (*polarity    literal     action  Cw  y   ->  p_reward *)
        let literal_f = X.expand_as(clauses).reshape([|1L;-1L|]).to_type(torch.int64)
        let action_f = torch.where(filter,os,zs).reshape([|1L;-1L|]).to_type(torch.int64)
        let cw_f = torch.hstack(ResizeArray[for _ in 1 .. int X.shape.[0] -> ce_t]).reshape([|1L;-1L|]).to_type(torch.int64)
        let y_f = y.expand_as(clauses).reshape([|1L;-1L|]).to_type(torch.int64)
        payoutMatrix.index([|polarityIndex; literal_f; action_f; cw_f; y_f|])

    //postive, negative or no feedback for each TA
    let taFeeback cfg (v:torch.Tensor) (pReward:torch.Tensor) (y:torch.Tensor) =
        use tPlus = torch.tensor(cfg.T, device=cfg.Device)
        use tMinus = torch.tensor(-cfg.T, device=cfg.Device)
        use t2 = (2.0f * cfg.T).ToScalar()
        use selY1 = (tPlus - v.clamp(tMinus,tPlus)) / t2  //feedback selection prob. when y=1
        use selY0 = (tPlus + v.clamp(tMinus,tPlus)) / t2  //feedback selection prob. when y=0 
        use uRand = torch.rand_like(pReward)              //tensor of uniform random values for Feedback selection
        use feebackFilter =                               //bool tensor; true if random <= [selY0 or selY1] (based on the value of y )
            if y.ToSingle() <= 0.f then 
                uRand.less_equal(selY0) 
            else 
                uRand.less_equal(selY1)                      
        use zeros = torch.zeros_like(pReward)                 //zero filled tensor - same shape as reward probabilities
        use pRewardSel = pReward.where(feebackFilter,zeros)   //keep reward prob if filter tensor value is true, zero otherwise
        use negRewards = pRewardSel.minimum(zeros)          //separate out negative reward prob.
        use posRewards = pRewardSel.maximum(zeros)          //separate out postive reward prob.
        use uRandRwrd = torch.rand_like(pRewardSel)         //tensor of uniform random values for Reward/Penalty selection
        use negRwdFltr = uRandRwrd.less_equal(negRewards.abs_()) //negative reward filter reflecting random selection
        use posRwdFltr = uRandRwrd.less_equal(posRewards)        //positive reward filter reflecting random selection
        use negOnes = torch.full_like(pRewardSel,-1)     //negative ones - will be used for negative feedback
        use posOnes = torch.full_like(pRewardSel, 1)     //positive ones - will be used for positive feedback
        use zerosInt = torch.zeros_like(pRewardSel)        
        use negFeedback = negOnes.where(negRwdFltr,zerosInt)
        use posFeedback = posOnes.where(posRwdFltr,zerosInt)
        negFeedback + posFeedback                                   //final feedback tensor with -1, +1 or 0 values 

    //calculate feedback incr/decr values based on selected feedback reward(+)/penalty(-)/ignore(0) and TA state
    let feedbackIncrDecr (cfg:Config) (clauses:torch.Tensor) (feedback:torch.Tensor) =
        use tMid = torch.tensor([|cfg.TAStates|], device=cfg.Device)
        use filterGreater = clauses.greater(tMid)
        use filterLessEqual = clauses.less_equal(tMid)
        use negOnes = torch.full_like(clauses,-1)     //negative ones - will be used for negative feedback
        use posOnes = torch.full_like(clauses, 1)     //positive ones - will be used for positive feedback
        let zeros   = torch.zeros_like(clauses)
        let highs = torch.where(filterGreater,posOnes,zeros)
        let lows = torch.where(filterLessEqual,negOnes,zeros)
        use feedbackPolarity = highs + lows
        feedbackPolarity.mul(feedback.reshape(clauses.shape))

    //inplace update clauses
    let updateClauses_ (cfg:Config) (clauses:torch.Tensor) (incrDecr:torch.Tensor) =
        let l = Scalar.op_Implicit 1
        let h = Scalar.op_Implicit (2*cfg.TAStates)
        clauses.add_(incrDecr).clamp_(l,h)

    //udpated clauses as a new tensor (for debugging purposes)
    let updateClauses (cfg:Config) (clauses:torch.Tensor) (incrDecr:torch.Tensor) =
        let l = Scalar.op_Implicit 1
        let h = Scalar.op_Implicit (2*cfg.TAStates)
        clauses.add(incrDecr).clamp_(l,h)

    //update clauses on single input - optimized for producton
    let trainStep cfg payoutMatrix (polarity,plrtySgn) clauses (X,y) = 
        use taEvals = Eval.evalTA cfg true clauses X //num_clauses * input
        use clauseEvals = Eval.andClause cfg taEvals
        use v = Eval.sumClauses cfg clauseEvals plrtySgn
        use pReward = rewardProb cfg payoutMatrix clauses clauseEvals polarity (X,y)
        use feedback = taFeeback cfg v pReward y
        use fbIncrDecr = feedbackIncrDecr cfg clauses feedback 
        updateClauses_ cfg clauses fbIncrDecr |> ignore

    //debug version of update that returns intermediate results
    let trainStepDbg cfg payoutMatrix (polarity,plrtySgn) clauses (X,y)  =
        let taEvals = Eval.evalTA cfg true clauses X  //num_clauses * input
        let clauseEvals = Eval.andClause cfg taEvals
        let v = Eval.sumClauses cfg clauseEvals plrtySgn
        let pReward = rewardProb cfg payoutMatrix clauses clauseEvals polarity (X,y)
        let feedback = taFeeback cfg v pReward y
        let fbIncrDecr = feedbackIncrDecr cfg clauses feedback 
        let updtClss = updateClauses cfg clauses fbIncrDecr 
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
        (*0           0           0       1   0 *)  inaction        //2
        (*0           0           0       1   1 *)  ``1/s``         //3
        (*0           0           1       0   0 *)  -``1/s``        //4
        (*0           0           1       0   1 *)  inaction        //5
        (*0           0           1       1   0 *)  inaction        //6
        (*0           0           1       1   1 *)  -``1/s``        //7
        (*0           1           0       0   0 *)  ``1/s``         //8
        (*0           1           0       0   1 *)  inaction        //9
        (*0           1           0       1   0 *)  inaction        //10
        (*0           1           0       1   1 *)  ``1/s``         //11
        (*0           1           1       0   0 *)  -``1/s``        //12
        (*0           1           1       0   1 *)  inaction        //13
        (*0           1           1       1   0 *)  inaction        //14
        (*0           1           1       1   1 *)  -``1/s``        //15
        (*1           0           0       0   0 *)  ``1/s``         //16
        (*1           0           0       0   1 *)  pnlty           //17
        (*1           0           0       1   0 *)  pnlty           //18
        (*1           0           0       1   1 *)  ``1/s``         //19
        (*1           0           1       0   0 *)  inaction        //20
        (*1           0           1       0   1 *)  inaction        //21
        (*1           0           1       1   0 *)  inaction        //22
        (*1           0           1       1   1 *)  inaction        //23
        (*1           1           0       0   0 *)  -``1/s``        //24
        (*1           1           0       0   1 *)  inaction        //25
        (*1           1           0       1   0 *)  inaction        //26
        (*1           1           0       1   1 *)  -``1/s``        //27
        (*1           1           1       0   0 *)  ``(s-1)/s``     //28
        (*1           1           1       0   1 *)  inaction        //29

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

        {
            Clauses         = clauses
            PolarityIndex   = polarityIdx
            PolaritySign    = torch.tensor(plrtySgn, dtype=cfg.dtype, device=cfg.Device)
            PayoutMatrix    = torch.tensor(payout cfg.s, dimensions = [|2L;2L;2L;2L;2L|], device=cfg.Device)   
            Config          = cfg
        }

    let train (X,y) (tm:TM) =
        Train.trainStep tm.Config tm.PayoutMatrix (tm.PolarityIndex,tm.PolaritySign) tm.Clauses (X,y)

    let trainBatch (X:torch.Tensor,y:torch.Tensor) (tm:TM) =
        let batchSze = X.shape.[0]
        for i in 0L .. batchSze - 1L do
            train (X.[i],y.[i]) tm

    let eval X (tm:TM) =
        use taEvals = Eval.evalTA tm.Config false tm.Clauses X //num_clauses * input
        use clauseEvals = Eval.andClause tm.Config taEvals
        use v = Eval.sumClauses tm.Config clauseEvals tm.PolaritySign
        if v.ToSingle() > 0.f then 1 else 0

    let evalBatch (X:torch.Tensor) (tm:TM) =
        let batchSze = X.shape.[0]
        [|for i in 0L .. batchSze - 1L do
            eval X.[i] tm
        |]

            
            
