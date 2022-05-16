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
            s                   = 3.0f
            T                   = 5.0f
            TAStates            = 6
            ClausesPerClass     = 4
            dtype               = torch.int16
            Device              = torch.CPU
            InputSize           = 2
            Classes             = 2
        }

    let tm = TM.create cfg

    let X = torch.tensor([|1;0;0;1|],dtype=cfg.dtype)     
    let v = TM.eval X tm
    Assert.Contains(v,[|0;1|])


[<Test>]
let TestTrain1 () =
    let cfg =
        {
            s                   = 3.0f
            T                   = 5.0f
            TAStates            = 6
            ClausesPerClass     = 4
            dtype               = torch.int16
            Device              = torch.CPU
            InputSize           = 2
            Classes             = 2
        }

    let tm = TM.create cfg

    let X = torch.tensor([|1;0;0;1|],dtype=cfg.dtype)     
    let y = torch.tensor([|0|],dtype=cfg.dtype)
    TM.train (X,y) tm


