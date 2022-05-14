module tsetlinTest

open NUnit.Framework
open TorchSharp
open FsTsetlin
open System.IO

[<SetUp>]
let Setup () =
    ()

[<Test>]
let TestEval () =
    let cfg =
        {
            s           = 3.0f
            T           = 5.0f
            TAStates    = 6
            Clauses     = 3
            dtype       = torch.int8
            Device      = torch.CPU
            InputSize   = 2
        }

    let tm = TM.create cfg

    let X = torch.tensor([|1;0;0;1|],dtype=torch.int8)     
    let v = TM.eval X tm
    Assert.AreEqual(v,0)


[<Test>]
let TestTrain1 () =
    let cfg =
        {
            s           = 3.0f
            T           = 5.0f
            TAStates    = 6
            Clauses     = 3
            dtype       = torch.int8
            Device      = torch.CPU
            InputSize   = 2
        }

    let tm = TM.create cfg

    let X = torch.tensor([|1;0;0;1|],dtype=torch.int8)     
    let y = torch.tensor([|0|],dtype=torch.int8)
    TM.train (X,y) tm


