﻿module Multiclass
//#load @"../scripts/packages.fsx"
open System.IO
open FsTsetlin
open TorchSharp

let trainDataFile = __SOURCE_DIRECTORY__ + @"/../data/BinaryIrisData.txt" 

let loadData (path:string) =
    File.ReadLines path
    |> Seq.map (fun l -> l.Split())
    |> Seq.map (fun xs -> xs |> Array.map int)
    |> Seq.map (fun xs -> 
        let x1 = xs.[0..^1]
        let X = Array.append x1 (x1 |> Array.map (function 1 -> 0 | 0 -> 1 | _ -> failwith "only 0 1 expected")) //with negated values
        let y = xs[^0]
        X,y)

let taStates (tm:TM) =
    let dt = tm.Clauses.``to``(torch.CPU).data<int32>().ToArray()
    dt |> Array.chunkBySize (tm.Invariates.Config.InputSize * 2)

let showClauses (tm:TM) =
    taStates tm
    |> Array.iteri (fun i x -> printfn "%d %A" i x)    

let trainData = loadData trainDataFile
let input = (trainData |> Seq.head |> fst |> Array.length)  / 2

let device = torch.CPU // if torch.cuda.is_available() then torch.CUDA else torch.CPU
printfn $"cuda: {torch.cuda.is_available()}"

let toTensor cfg (batch:(int[]*int)[]) =
    let batchSize = int64 batch.Length
    let X = torch.tensor(batch |> Array.collect fst, dtype = cfg.dtype, device = cfg.Device, dimensions = [| batchSize ; cfg.InputSize*2 |> int64|])
    let y = torch.tensor(batch |> Array.map snd, dtype = torch.int64, device = cfg.Device, dimensions = [| batchSize; 1L |> int64|])
    X,y

let cfg =
    {
        s           = 3.0f
        T           = 10.0f
        TAStates    = 100
        dtype       = torch.int32
        Device      = device
        InputSize   = input
        ClausesPerClass = 100
        Classes         = 3
    }

let tm = TM.create cfg

let eval() =
    trainData
    |> Seq.chunkBySize 1000
    |> Seq.map (toTensor tm.Invariates.Config)
    |> Seq.collect (fun (X,y) -> 
        [for i in 0L .. X.shape.[0] - 1L do
            yield TM.predict X.[i] tm, y.[i].ToInt32()
        ])
    |> Seq.map (fun (y',y) -> if y' = y then 1.0 else 0.0)
    |> Seq.average

let train epochs =
    for i in 1 .. epochs do
        trainData
        |> Seq.chunkBySize 1000
        |> Seq.map (toTensor tm.Invariates.Config)
        |> Seq.iter (fun (X,y) -> 
            TM.trainBatch (X,y) tm
            X.Dispose()
            y.Dispose())
        printfn $"{i}: {eval()}"

let run() =
    train 1
;;


(*
showClauses tm

let tas = taStates tm
tas |> Array.map(fun xs -> xs |> Array.indexed |> Chart.Line) |> Chart.combine |> Chart.show

*)




