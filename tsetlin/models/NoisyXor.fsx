#load @"../scripts/packages.fsx"
#load @"../Tsetlin.fs"
open System.IO
open FsTsetlin
open TorchSharp

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
        MidState    = 100/2
        Clauses     = 20
        dtype       = torch.int8
        Device      = device
        InputSize   = 12
    }

let tm = TM.create cfg

let train epochs =
    for i in 1 .. epochs do
        trainData
        |> Seq.chunkBySize 100 
        |> Seq.map (toTensor cfg)
        |> Seq.iter (fun (X,y) -> 
            TM.trainBatch (X,y) tm
            X.Dispose()
            y.Dispose())

train 1

       