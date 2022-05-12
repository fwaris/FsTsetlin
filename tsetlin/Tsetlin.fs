namespace FsTsetlin 
open TorchSharp

module Tsetline =
    let TA_MIN = 1uy
    let TA_MAX = 6uy
    let TA_MID = 3uy

    module Eval = 
        
        //eval each literal against each tsetlin automaton (TA) in the clauses
        let evalTA trainMode (input:torch.Tensor) (clauses:torch.Tensor) = 
            let input2 = input.broadcast_to(clauses.shape)
            use t3 = torch.tensor(TA_MID)
            use filter = clauses.greater(t3)
            use omask = if trainMode then torch.ones_like(input2) else torch.zeros_like(input2)   //default to 1 for training and 0 for evaluation
            torch.where(filter,input2,omask)

        //AND the outputs of TAs by clause
        let andClause (evals:torch.Tensor)  =
            use prods = evals.cumprod(1L,``type``=torch.int8)
            prods.[torch.TensorIndex.Ellipsis,torch.TensorIndex.Single(-1L)]

        //sum positive and negative polarity clause outputs
        let sumClauses (clauseEvals:torch.Tensor) (plrtySgn:torch.Tensor) =
            use withPlry = clauseEvals.mul(plrtySgn)
            withPlry.sum().to_type(torch.float32)


    module Train = 
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

        //obtain +/- reward probabilities for each TA (for type I and II feedback)
        let rewardProb (payoutMatrix:torch.Tensor) (input:torch.Tensor) (clauses:torch.Tensor) (clauseEvals:torch.Tensor) (polarity:torch.Tensor) (y:torch.Tensor) =
            use t3 = torch.tensor([|TA_MID|])
            use filter = clauses.greater(t3)
            use zs = torch.zeros_like(clauses)
            use os = torch.ones_like(clauses)
            let ce_t = clauseEvals.reshape(-1L,1L)
            (*polarity    literal     action  Cw  y   ->  p_reward *)
            let polarity_f = polarity.expand_as(clauses).reshape([|1L;-1L|]).to_type(torch.int64)
            let literal_f = input.expand_as(clauses).reshape([|1L;-1L|]).to_type(torch.int64)
            let action_f = torch.where(filter,os,zs).reshape([|1L;-1L|]).to_type(torch.int64)
            let cw_f = torch.hstack(ResizeArray[for _ in 1 .. int input.shape.[0] -> ce_t]).reshape([|1L;-1L|]).to_type(torch.int64)
            let y_f = y.expand_as(clauses).reshape([|1L;-1L|]).to_type(torch.int64)
            payoutMatrix.index([|polarity_f; literal_f; action_f; cw_f; y_f|])

        //postive, negative or no feedback for each TA
        let taFeeback (T:float32) (v:torch.Tensor) (pReward:torch.Tensor) (y:torch.Tensor) =
            use tPlus = torch.tensor(T)
            use tMinus = torch.tensor(-T)
            use t2 = (2.0f * T).ToScalar()
            use selY1 = (tPlus - v.clamp(tMinus,tPlus)) / t2  //feedback selection prob. when y=1
            use selY0 = (tPlus + v.clamp(tMinus,tPlus)) / t2  //feedback selection prob. when y=0 
            use uRand = torch.rand_like(pReward)              //tensor of uniform random values (for feedback selection)
            use feebackFilter =                               //bool tensor; true if random <= [selY0 or selY1] (based on the value of y )
                if y.ToSingle() <= 0.f then 
                    uRand.less_equal(selY0) 
                else 
                    uRand.less(selY1)                      
            use zeros = torch.zeros_like(pReward)                 //zero filled tensor - same shape as reward probabilities
            use pRewardSel = pReward.where(feebackFilter,zeros)   //keep reward prob if filter tensor value is true, zero otherwise
            use negRewards = pRewardSel.minimum(zeros)          //separate out negative reward prob.
            use posRewards = pRewardSel.maximum(zeros)          //separate out postive reward prob.
            use uRandRwrd = torch.rand_like(pRewardSel)         //tensor of uniform random values for reward selection
            use negRwdFltr = uRandRwrd.less_equal(negRewards.abs_()) //negative reward filter reflecting random selection
            use posRwdFltr = uRandRwrd.less_equal(posRewards)        //positive reward filter reflecting random selection
            use negOnes = torch.full_like(pRewardSel,-1, dtype=torch.uint8)     //negative ones - will be used for negative feedback
            use posOnes = torch.full_like(pRewardSel, 1, dtype=torch.uint8)     //positive ones - will be used for positive feedback
            use zerosI8 = torch.zeros_like(pRewardSel,dtype=torch.uint8)        
            use negFeedback = negOnes.where(negRwdFltr,zerosI8)
            use posFeedback = posOnes.where(posRwdFltr,zerosI8)
            use feedbackByte = negFeedback + posFeedback
            feedbackByte.to_type(torch.int8)                                 //final feedback tensor with -1, +1 or 0 values 

        //inplace update clauses
        let private taMinScalar = TA_MIN.ToScalar()
        let private taMaxScalar = TA_MAX.ToScalar()
        let updateClauses_ (clauses:torch.Tensor) (feedback:torch.Tensor) =
            clauses.add_(feedback.reshape(clauses.shape)).clamp_(taMinScalar,taMaxScalar)

        //udpated clauses as a new tensor
        let updateClauses (clauses:torch.Tensor) (feedback:torch.Tensor) =
            clauses.add(feedback.reshape(clauses.shape)).clamp_(taMinScalar,taMaxScalar)

        //update clauses on single input - optimized for producton
        let trainStep  T payoutMatrix (polarity,plrtySgn) clauses (input,y) = 
            use taEvals = Eval.evalTA true input clauses //num_clauses * input
            use clauseEvals = Eval.andClause taEvals
            use v = Eval.sumClauses clauseEvals plrtySgn
            use pReward = rewardProb payoutMatrix input clauses clauseEvals polarity y
            use feedback = taFeeback T v pReward y
            updateClauses_ clauses feedback |> ignore

        //debug version of update that returns intermediate results
        let trainStepDbg T payoutMatrix (polarity,plrtySgn) clauses (input,y)  =
            let taEvals = Eval.evalTA true input clauses //num_clauses * input
            let clauseEvals = Eval.andClause taEvals
            let v = Eval.sumClauses clauseEvals plrtySgn
            let pReward = rewardProb payoutMatrix input clauses clauseEvals polarity y
            let feedback = taFeeback T v pReward y
            let updtClss = updateClauses clauses feedback 
            taEvals,clauseEvals,v,pReward,feedback,updtClss
