﻿#load @"../scripts/packages.fsx"
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
    testData
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
        

let tX1,ty1 = [|trainData |> Seq.item 3|] |> toTensor tm.Invariates.Config
let X1=tX1.squeeze()
let y1=ty1.squeeze()

let taEvals,clauseEvals,v,pReward,feedback,fbIncrDecr,updtClss = Train.trainStepDbg tm.Invariates tm.Clauses (X1,y1)

let s = 3.9f
let ``1/s``     = 1.0f / s
let ``(s-1)/s`` = (s - 1.0f) / s
let ( =^ ) a b = abs (a - b) < 0.000001f
let mapProb x = 
    if x =^ ``1/s`` then sprintf "``1/s``; " 
    elif x =^ ``(s-1)/s`` then sprintf "``(s-1)/s``; " 
    elif x =^ -``1/s`` then sprintf "-``1/s``; " 
    elif x =^ -``(s-1)/s`` then sprintf "-``(s-1)/s``; " 
    else sprintf "%f" x

let printClause i =
    // let i = 1
    let tas         = Utils.tensorData<int16> taEvals
    let clauseEvals = Utils.tensorData<int16> clauseEvals
    let input       = Utils.tensorData<int16> X1
    let ty          = Utils.tensorData<int16> y1
    let rewards     = Utils.tensorData<float32> (pReward.reshape(tm.Clauses.shape))
    let feeback     = Utils.tensorData<int16> (feedback.reshape(tm.Clauses.shape))
    let clss        = Utils.tensorData<int16> tm.Clauses
    let fbIncrDecr  = Utils.tensorData<int16> fbIncrDecr
    let updClss     = Utils.tensorData<int16> updtClss
    let polarity    = Utils.tensorData<int64> (tm.Invariates.PolarityIndex.reshape(tm.Clauses.shape))

    let act = match tas with Utils.T (ds) -> match ds.[i] with Utils.F xs -> xs 
    let prob = match rewards with Utils.T ds -> match ds.[i] with Utils.F xs -> xs
    let inp1 = match input with Utils.F xs -> xs
    let clsout = match clauseEvals with Utils.F xs -> xs.[i]
    let y = match ty with Utils.F xs -> xs.[0]
    let w = match polarity with Utils.T ds -> match ds.[i] with Utils.F xs -> xs.[0]
    let fb = match feeback with Utils.T ds -> match ds.[i] with Utils.F xs -> xs
    let fbid = match fbIncrDecr with Utils.T ds -> match ds.[i] with Utils.F xs -> xs
    let clssA = match clss with Utils.T ds -> match ds.[i] with Utils.F xs -> xs
    let updClssA = match updClss with Utils.T ds -> match ds.[i] with Utils.F xs -> xs

    for j in 0 .. act.Length-1 do
        let a = act.[j]
        let p = prob.[j]
        let l = inp1.[j]
        let fb = fb.[j]
        let incDec = fbid.[j]
        printfn $"C:{clsout}, y:{y}, w:{w}, L:{l}, act:{(if a = 0s then 'i' else 'x')}, pReward: {mapProb p}, fb: {fb}, incr/decr: {incDec}, clssIn:{clssA.[j]}, clssOut:{updClssA.[j]}"

printClause 0
printClause 1
printClause 2
printClause 3
printClause 4

let print

(*
#time

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
 
let rawProbs = [|0.2564102411f; 0.2564102411f; 0.2564102411f; -0.2564102411f;
                  -0.2564102411f; -0.2564102411f; 0.2564102411f; -0.2564102411f;
                  0.2564102411f; 0.2564102411f; -0.2564102411f; -0.2564102411f;
                  0.2564102411f; 0.2564102411f; -0.2564102411f; 0.2564102411f;
                  0.2564102411f; -0.2564102411f; 0.2564102411f; 0.2564102411f;
                  0.2564102411f; 0.2564102411f; 0.2564102411f; 0.2564102411f|];
rawProbs |> Array.iter (mapProb>>(printf "%s"))
rawProbs.Length


let taS  =   [|100s; 100s; 100s; 101s; 101s; 101s; 100s; 101s; 100s; 100s; 101s; 101s; 100s; 100s; 101s; 100s; 100s; 101s; 100s; 100s; 100s; 100s; 100s; 100s|]
//let act = taS |> Array.map (fun x -> if x > 100s then 1s else 0s)
let inp1 =  [|0s; 1s; 1s; 0s; 0s; 0s; 0s; 1s; 1s; 1s; 1s; 0s; 1s; 0s; 0s; 1s; 1s; 1s; 1s; 0s; 0s; 0s; 0s; 1s|]
let act  =  [|0s; 0s; 0s; 1s; 1s; 1s; 0s; 1s; 0s; 0s; 1s; 1s; 0s; 0s; 1s; 0s; 0s; 1s; 0s; 0s; 0s; 0s; 0s; 0s|]
let clsout = 0
let y = 0
let w = 0
let prob = [|``1/s``; ``1/s``; ``1/s``; -``1/s``; -``1/s``; -``1/s``; ``1/s``; -``1/s``; ``1/s``; ``1/s``; -``1/s``; -``1/s``; ``1/s``; ``1/s``; -``1/s``; ``1/s``; ``1/s``; -``1/s``; ``1/s``; ``1/s``; ``1/s``; ``1/s``; ``1/s``; ``1/s``|]


*)