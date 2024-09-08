module FsTsetlinTest

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
        { Config.Default with
            s                   = 3.0f
            T                   = 5.0f
            TAStates            = 6
            ClausesPerClass     = 4
            Device              = torch.CPU
            InputSize           = 2
            Classes             = 2
        }

    let tm = TM.create cfg

    let X = torch.tensor([|1;0;0;1|],dtype=torch.int8)     
    let v = TM.predict X tm
    let vValid = Array.contains v [|0;1|]
    Assert.That(vValid,"predict failed")


[<Test>]
let TrainBinary () =
    let cfg =
        { Config.Default with
            s                   = 3.0f
            T                   = 5.0f
            TAStates            = 6
            ClausesPerClass     = 4
            Device              = torch.CPU
            InputSize           = 2
            Classes             = 2
        }

    let tm = TM.create cfg

    let X = torch.tensor([|1;0;0;1|],dtype=torch.int8)     
    let y = torch.tensor([|0|],dtype=torch.int8)
    TM.train (X,y) tm

[<Test>]
let TrainMulti () =
    let cfg =
        { Config.Default with
            s                   = 3.0f
            T                   = 5.0f
            TAStates            = 6
            ClausesPerClass     = 4
            Device              = torch.CPU
            InputSize           = 2
            Classes             = 3
        }

    let tm = TM.create cfg

    let X = torch.tensor([|1;0;0;1|],dtype=torch.int8)     
    let y = torch.tensor([|0|],dtype=torch.int8)
    TM.train (X,y) tm

[<Test>]
let LoadSave () =
    let cfg =
        { Config.Default with
            s                   = 3.0f
            T                   = 5.0f
            TAStates            = 6
            ClausesPerClass     = 4
            Device              = torch.CPU
            InputSize           = 2
            Classes             = 3
        }

    let tm = TM.create cfg

    let X = torch.tensor([|1;0;0;1|],dtype=torch.int8)     
    let y = torch.tensor([|2|],dtype=torch.int8)
    TM.train (X,y) tm
    let cls1,wts1 = TM.exportLearned tm
    let fn = System.IO.Path.GetTempFileName()
    TM.save fn tm
    let tm2 = TM.load torch.CPU fn
    let cls2,wts2 = TM.exportLearned tm2
    Assert.That( (cls1 = cls2) , message = "clauses states not match")
    Assert.That( (wts1 = wts2) , message = "clauses weights not match")

    TM.train (X,y) tm2
    ()


