namespace FsTsetlin
open TorchSharp
open System

module Utils =
    type D<'a> = 
        | A of 'a[]         // flat array of values - from the inner most dimension
        | G of D<'a>[]      // group of groups or flat arrays

    //utility function to get raw tensor data as a recursive structure for debugging purposes
    let getDataNested<'a when 'a: unmanaged and  'a: (new: unit -> 'a) and  'a: struct and 'a :> ValueType>(t:torch.Tensor) = 
        let ts = if t.device<>torch.CPU then t.cpu().data<'a>().ToArray() else t.data<'a>().ToArray()
        let rdims =
            t.shape 
            |> Array.map int 
            |> Array.rev            //start with inner most dimension
            |> Array.toList
        let rec loop ds (xs:D<'a>) =
            match ds,xs with
            | [],_                        -> xs
            | d::[],G ds when d=ds.Length -> G ds
            | d::[],A ds when d=ds.Length -> A ds
            | d::rest,G ds -> loop rest (ds |> Array.chunkBySize d |> Array.map G |> G)
            | d::rest,A ds -> loop rest (ds |> Array.chunkBySize d |> Array.map A |> G)
        loop rdims (A ts)
