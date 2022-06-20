module mnist
open System
open System.IO

let loadData (path:string) =
    File.ReadLines path
    |> Seq.map (fun l -> l.Split())
    |> Seq.map (fun xs -> xs |> Array.map int)
    |> Seq.map (fun xs -> 
        let x1 = xs.[0..xs.Length-2]
        let X = Array.append x1 (x1 |> Array.map (function 1 -> 0 | 0 -> 1 | _ -> failwith "only 0 1 expected")) //with negated values
        let y = xs[xs.Length-1]
        X,y)

let mnistTrainFile = __SOURCE_DIRECTORY__ + @"/../data/BinarizedMNISTData/MNISTTraining.txt"
let mnistTestFile = __SOURCE_DIRECTORY__ + @"/../data/BinarizedMNISTData/MNISTTest.txt"
let mnistTrain = loadData mnistTrainFile
let mnistTest = loadData mnistTestFile

open TorchSharp
open FsTsetlin
let toTensor cfg  (batch:(int[]*int)[]) =
    let batchSize = int64 batch.Length
    let xs = batch |> Array.collect fst 
    let X = torch.tensor(xs, dtype = torch.int8, device = cfg.Device, dimensions = [| batchSize ; cfg.InputSize*2 |> int64|])
    let y = torch.tensor(batch |> Array.map snd, dtype = torch.int64, device = cfg.Device, dimensions = [| batchSize; 1L |> int64|])
    X,y

//let lt = loadData mnistTrainFile |> Seq.toArray

//let ft = lt |> Array.tryFind (fun (xs,y) -> xs |> Array.exists (fun y-> (y=0 || y=1) |> not))

let device = if torch.cuda.is_available() then torch.CUDA else torch.CPU
let inputSize = ((Seq.head mnistTrain) |> fst |> Array.length) / 2
let classes = (mnistTrain |> Seq.map snd) |> Seq.distinct |> Seq.length
let cfg =
    { Config.Default with
        s           = 3.0f
        T           = 10.0f
        TAStates    = 100       
        Device      = device
        InputSize   = inputSize
        ClausesPerClass = 1000 //total = 10 * 100 = 1000
        Classes         = classes
        MaxWeight       = 1
    }

let tm = TM.create cfg
printfn $"device : {device}, inputSize: {inputSize}, classes: {classes}"

let eval() =
    mnistTest
    |> Seq.chunkBySize 1000
    |> Seq.map (toTensor tm.TMState.Config)
    |> Seq.collect (fun (X,y) -> 
        [for i in 0L .. X.shape.[0] - 1L do
            yield TM.predict X.[i] tm, y.[i].ToInt32()
        ])
    //|> Seq.map snd |> Seq.countBy (fun x->x) |> Seq.toArray
    |> Seq.map (fun (y',y) -> if y' = y then 1.0 else 0.0)
    //|> Seq.toArray |> Array.distinct
    |> Seq.average

let train epochs =
    for i in 1 .. epochs do
        mnistTrain
        |> Seq.chunkBySize 30000
        |> Seq.map (toTensor tm.TMState.Config)
        |> Seq.iter (fun (X,y) -> 
            TM.trainBatch (X,y) tm
            X.Dispose()
            y.Dispose()) 

let trainTimed i =
    let tstart = DateTime.Now
    train 1
    let tend = DateTime.Now
    let elapsed = (tend - tstart).TotalMinutes
    let acc = eval()
    printfn $"run: {i}, elapsed minutes: {elapsed}, accuracy: {acc}"
    elapsed,acc
            
let runTrain i =
    for j in 1 .. i do 
        trainTimed j |> ignore

let runner() = 
    async {
        let results = [for i in 1 .. 10 -> trainTimed i ]
        let avgTime = results |> List.map fst |> List.average
        printf $"average time minutes: {avgTime}"    
    }

(*
runner() |> Async.Start

*)
let run() = runTrain 1


