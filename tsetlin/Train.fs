namespace FsTsetlin 
open TorchSharp

module Train =
    let inaction = 0.0f
    let pnlty    = -1.0f

    let payout s = 
        let ``1/s``     = 1.0f / s
        let ``(s-1)/s`` = (s - 1.0f) / s
        [|
        (*polarity    literal     action  Cw  y     p_reward *)-
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

    

