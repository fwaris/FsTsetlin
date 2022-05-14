#load @"../scripts/packages.fsx"
open System
open System.IO
open FsTsetlin
open TorchSharp
open Plotly.NET

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

let taStates invariates (clauses:torch.Tensor)=
    let dt = clauses.``to``(torch.CPU).data<int16>().ToArray()
    dt |> Array.chunkBySize (invariates.Config.InputSize * 2)

let showClauses invariates clauses =
    taStates invariates clauses
    |> Array.iteri (fun i x -> printfn "%d %A" i x)    

let trainData = loadData trainDataFile
let testData = loadData testDataFile

let device = torch.CPU // if torch.cuda.is_available() then torch.CUDA else torch.CPU
printfn $"cuda: {torch.cuda.is_available()}"

let toTensor cfg (batch:(int[]*int)[]) =
    let batchSize = int64 batch.Length
    let X = torch.tensor(batch |> Array.collect fst, dtype = cfg.dtype, device = cfg.Device, dimensions = [| batchSize ; cfg.InputSize*2 |> int64|])
    let y = torch.tensor(batch |> Array.map snd, dtype = cfg.dtype, device = cfg.Device, dimensions = [| batchSize; 1L |> int64|])
    X,y

let cfg =
    {
        s           = 3.9f
        T           = 15.0f
        TAStates    = 100
        Clauses     = 5
        dtype       = torch.int16
        Device      = device
        InputSize   = 12
    }

let tm = TM.create cfg

let eval() =
    trainData
    |> Seq.chunkBySize 1000
    |> Seq.map (toTensor tm.Invariates.Config)
    |> Seq.collect (fun (X,y) -> 
        [for i in 0L .. X.shape.[0] - 1L do
            yield TM.eval X.[i] tm, y.[i].ToInt32()
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
        showClauses tm.Invariates tm.Clauses
#time

let X1,y1 = [|trainData |> Seq.head|] |> toTensor tm.Invariates.Config

let taEvals,clauseEvals,v,pReward,feedback,fbIncrDecr,updtClss = Train.trainStepDbg tm.Invariates tm.Clauses (X1.squeeze(),y1.squeeze())

showClauses tm.Invariates tm.Clauses
showClauses tm.Invariates updtClss

Utils.tensorData<int16> taEvals
Utils.tensorData<int16> clauseEvals
Utils.tensorData<int16> X1
Utils.tensorData<int16> y1
Utils.tensorData<float32> (pReward.reshape(tm.Clauses.shape))
Utils.tensorData<int16> (feedback.reshape(tm.Clauses.shape))
Utils.tensorData<int16> fbIncrDecr
Utils.tensorData<int16> updtClss




train 1

(*

let tas = taStates tm
tas |> Array.map(fun xs -> xs |> Array.indexed |> Chart.Line) |> Chart.combine |> Chart.show

//utility function to get raw tensor data as a flat array (shape is not retained)
let tensorData<'a when 'a: (new: unit -> 'a) and  'a: struct and 'a :> ValueType>(t:torch.Tensor) = t.data<'a>().ToArray()


let t1 = torch.tensor([|0;1;3;4|],dimensions=[|2L;2L|])
let t2 = t1.greater(2)
tensorData<bool> t2

let t3 = t2.any(1L)
tensorData<bool> t3

let t4 = t2.any(1L,keepDim=false)
tensorData<bool> t4

*)
