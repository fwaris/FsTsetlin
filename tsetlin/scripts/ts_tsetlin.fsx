#load "packages.fsx"

open TorchSharp
open type TorchSharp.torch.nn

let ``:`` = torch.TensorIndex.Colon
let first = torch.TensorIndex.Single(0L)
let last = torch.TensorIndex.Single(-1L)

module Inference =
    //inference
    let t3 = torch.tensor(3uy)// TorchSharp.Scalar.op_Implicit(3uy)

    let inp = torch.tensor([|1;0;0;1|],dtype=torch.int8)         //input x's and ~x's
    let tas = torch.tensor([|1;5;6;2|],dtype=torch.int8)         //tas states
    let out = torch.tensor([|1;0;0;1|],dtype=torch.int8)         //expected output

    let filter = tas.greater(t3) //bools, true if tas > 3, false otherwise
    let omsk = torch.tensor([|1;1;1;1|],dtype=torch.int8)        //output 1's if filter true for training; changes to 0's for inference
    let out' = torch.where(filter,inp,omsk)
    let cnf = out'.cumprod(0L,``type``=torch.int8) // conjunction
    let clauseOut = cnf.[3] |> int8 //clause output is last element

    cnf.data<int8>().ToArray()
    out'.data<int8>().ToArray()
    out.data<int8>().ToArray()

module Type1 = 
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
    let ``1/s``     = 1.0f / s
    let ``(s-1)/s`` = (s - 1.0f) / s
    let rewardProb = 
        [|
            //polarity, literal, action, Cw, y
            (* 0,0 0,0,0 *)         ``1/s``
            (* 0,0,0,0,1 *)         
            (* 0,0,1 *)  -``1/s``
            (* 0,1,0 *)  ``1/s``
            (* 0,1,1 *)  -``1/s``
            (* 1,0,0 *)  ``1/s``
            (* 1,0,1 *)  1.f
            (* 1,1,0 *)  -``1/s``
            (* 1,1,1 *)  ``(s-1)/s``
        |]

    let t_probs = torch.tensor(type1Prob,dimensions=[|2L;2L;2L|])




    let t3 = torch.tensor(3uy)// TorchSharp.Scalar.op_Implicit(3uy)

    let inp = torch.tensor([|0;1;1;0|],dtype=torch.int8)         //input x's and ~x's
    let tas = torch.tensor([|1;5;6;2
                             1;3;2;6|],dimensions=[|2L;4L|],dtype=torch.int8)         //2 tas states
    let out = torch.tensor([|1;0;0;1|],dtype=torch.int8)                              //expected output

    let filter = tas.greater(t3) //bools, true if tas > 3, false otherwise  filter.data<bool>().ToArray()
    let omsk = torch.ones_like(tas)  
    let out' = torch.where(filter,inp,omsk)                                 // out'.data<int8>().ToArray()
    let cnf = out'.cumprod(1L,``type``=torch.int8) // conjunction cnf.data<int8>().ToArray()
    let clauseOut = cnf.index(``:``,last)  //clauseOut.data<int8>().ToArray()

    let y = torch.ones_like(clauseOut)




    // assume positive clause with y = 1

    let s = 3.0
    let type1Probs =
        //dimensions: polarity, y, w, literal, action, preward/penalty
        //polarity, y, Cw, literal, action -> preward/penalty
        [|
            
            
        |]

    let e = torch.nn.em
    (*
    - for each TA compare y with Cw 
    - if y = Cw then type 1 else type 2 
    - compare ta action with corresponding literal to determine prob. for:
      p reward, p penalty, p inaction 
    - fill reward matrix for TA using reward probs    (0 for no action)
    - fill penalty matrix for TA using penalty probs  (0 for no action)
    - apply reward matrix
    - apply penalty matrix
    - alterative: single reward/penalty matrix
      +1 for reward; -1 for penalty; 0 for nothing
    *)





(*
type 2 feedback table y<>w
c      literal  include 1   p reward      p penalty       p inaction
+/-    value    exclude 0                                 (dont' change)
+      l = 1    a = 1       0              0               1.0
+      l = 0    a = 1       NA             NA              NA              //situation does not occur
+      l = 1    a = 0       0              0               1.0
+      l = 0    a = 0       0              1.0             0

-      l = 1    a = 1       0              0               1.0
-      l = 0    a = 1       0              0               1.0
-      l = 1    a = 0       0              0               1.0
-      l = 0    a = 0       0              0               1.0
*)

