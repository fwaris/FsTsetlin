#load @"../scripts/packages.fsx"
#r "nuget: FSharp.Control.AsyncSeq"

let LIBTORCH = 
    let path = System.Environment.GetEnvironmentVariable("LIBTORCH")
    if path <> null then path else @"D:\s\libtorch\lib\torch_cuda.dll"
System.Runtime.InteropServices.NativeLibrary.Load(LIBTORCH) |> ignore
let path = System.Environment.GetEnvironmentVariable("path")
let path' = $"{path};{LIBTORCH}"
System.Environment.SetEnvironmentVariable("path",path')

open System.IO
open FsTsetlin
open TorchSharp
open Plotly.NET
open FSharp.Control

let testDataFile = __SOURCE_DIRECTORY__ + @"/../../data/NoisyXORTestData.txt" 
let trainDataFile = __SOURCE_DIRECTORY__ + @"/../../data/NoisyXORTrainingData.txt" 

let loadData (path:string) =
    File.ReadLines path
    |> Seq.map (fun l -> l.Split())
    |> Seq.map (fun xs -> xs |> Array.map int)
    |> Seq.map (fun xs -> 
        let x1 = xs.[0..11]
        let X = Array.append x1 (x1 |> Array.map (function 1 -> 0 | 0 -> 1 | _ -> failwith "only 0 1 expected")) //with negated values
        let y = xs[12]
        X,y)

let taStates (tm:TM) =
    let dt = tm.Clauses.``to``(torch.CPU).data<int32>().ToArray()
    dt |> Array.chunkBySize (tm.TMState.Config.InputSize * 2)

let showClauses (tm:TM) =
    taStates tm
    |> Array.iteri (fun i x -> printfn "%d %A" i x)    

let trainData() = loadData trainDataFile
let testData() = loadData testDataFile

let device = if torch.cuda.is_available() then torch.CUDA else torch.CPU
printfn $"cuda: {torch.cuda.is_available()}"

let toTensor cfg (batch:(int[]*int)[]) =
    let batchSize = int64 batch.Length
    let X = torch.tensor(batch |> Array.collect fst, dtype = torch.int8, device = cfg.Device, dimensions = [| batchSize ; cfg.InputSize*2 |> int64|])
    let y = torch.tensor(batch |> Array.map snd, dtype = torch.int64, device = cfg.Device, dimensions = [| batchSize; 1L |> int64|])
    X,y

let cfg =
    { Config.Default with
        s           = 3.9f
        T           = 15.0f
        TAStates    = 100
        Device      = device
        InputSize   = 12
        ClausesPerClass = 1000
        Classes         = 2
    }

let tm = TM.create cfg

let eval() =
    testData()
    |> Seq.chunkBySize 10000
    |> Seq.map (toTensor tm.TMState.Config)
    |> Seq.collect (fun (X,y) -> 
        [for i in 0L .. X.shape.[0] - 1L do
            yield TM.predict X.[i] tm, y.[i].ToInt32()
        ])
    |> Seq.map (fun (y',y) -> if y' = y then 1.0 else 0.0)
    |> Seq.average

let trainEpoch() = 
        trainData()
        |> Seq.chunkBySize 10000 
        |> AsyncSeq.ofSeq
        |> AsyncSeq.map (toTensor tm.TMState.Config)
        |> AsyncSeq.map  (fun (X,y) -> 
            TM.trainBatch (X,y) tm
            X,y)
        |> AsyncSeq.iter (fun (X,y) ->
            X.Dispose()
            y.Dispose())
open System
let train epochs =    
    async {
        let t1 = DateTime.Now
        for i in 1 .. epochs do
            do! trainEpoch()
            let acc = eval()
            printfn $"Epoch: {i}, Acc:{acc}"
        let t2 = DateTime.Now
        let elapsed = (t2-t1).TotalMinutes
        printfn $"time {elapsed}"
    }
    |> Async.Start


#time
(*
train 10
*)
;;


(*
showClauses tm

let tas = taStates tm
tas |> Array.map(fun xs -> xs |> Array.indexed |> Chart.Line) |> Chart.combine |> Chart.show

*)


