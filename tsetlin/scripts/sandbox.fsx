#load "packages.fsx"

open TorchSharp
    (*
    type 1 feedback table y=w
    c      literal  include 1   p reward      p penalty       p inaction
    +/-    value    exclude 0                                 (don't change)
    +      l = 1    a = 1       (s-1)/s        NA              1/s
    +      l = 0    a = 1       NA             NA              NA              //situation does not occur
    +      l = 1    a = 0       0              1/s             (s-1)/s  
    +      l = 0    a = 0       1/s            0               (s-1)/s 

    -      l = 1    a = 1      0              1/s              (s-1)/s
    -      l = 0    a = 1      0              1/s              (s-1)/s
    -      l = 1    a = 0      1/s            0                (s-1)/s
    -      l = 0    a = 0      1/s            0                (s-1)/s
    *)
let s = 3.0f
let ``1/s``     = 1.0f / 3.0f
let ``(s-1)/s`` = (s - 1.0f) / s

let type1Prob = 
    [|
        //polarity, literal, action 
        (* 0,0,0 *)  ``1/s``
        (* 0,0,1 *)  -``1/s``
        (* 0,1,0 *)  ``1/s``
        (* 0,1,1 *)  -``1/s``
        (* 1,0,0 *)  ``1/s``
        (* 1,0,1 *)  1.f
        (* 1,1,0 *)  -``1/s``
        (* 1,1,1 *)  ``(s-1)/s``
    |]

let t_probs = torch.tensor(type1Prob,dimensions=[|2L;2L;2L|])

let p_idx = [|true; true; true|]
let p_r = t_probs.index(true,true,true)
p_r.data<float32>().ToArray()
let pr2 = t_probs.index(p_idx)

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

let rng = System.Random()
let x = rng.NextDouble()
let mv = System.Double.MaxValue
let j = x / mv

let target = 15.
let t1 = (2.* target) - 1.0
let j = 3
let jsgn = 1 - 2 * (j &&& 1)
