﻿#load "packages.fsx"

open TorchSharp
let src = 
    [|
        //polarity, literal, action 
        (* 0,0,0 *)  1.f
        (* 0,0,1 *)  2.f
        (* 0,1,0 *)  3.f
        (* 0,1,1 *)  4.f
        (* 1,0,0 *)  5.f
        (* 1,0,1 *)  6.f
        (* 1,1,0 *)  7.f
        (* 1,1,1 *)  8.f
    |]
let tsrc = torch.tensor(src,dimensions=[|2L;2L;2L|])
let i_polarity = torch.tensor([|0L;1L|])
let i_literal  = torch.tensor([|0L;1L|])
let i_action   = torch.tensor([|0L;1L|])
let t = tsrc.index([|i_polarity;i_literal;i_action|])
t.data<float32>().ToArray()

let t2 = FsTsetlin.Utils.tensorData<float32> tsrc


let rng = System.Random()
let x = rng.NextDouble()
let mv = System.Double.MaxValue
let j = x / mv

let target = 15.
let t1 = (2.* target) - 1.0
let j = 3
let jsgn = 1 - 2 * (j &&& 1)
