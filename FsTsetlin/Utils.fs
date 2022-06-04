namespace FsTsetlin
open TorchSharp
open System

module Utils =
    type D<'a> = F of 'a[] | T of D<'a>[]

    //utility function to get raw tensor data as a recursive structure for debugging purposes
    let tensorData<'a when 'a: (new: unit -> 'a) and  'a: struct and 'a :> ValueType>(t:torch.Tensor) = 
        let ts = if t.device<>torch.CPU then t.cpu().data<'a>().ToArray() else t.data<'a>().ToArray()
        let rdims =
            t.shape 
            |> Array.map int 
            |> Array.rev            //start with inner most dimension
            |> Array.toList
        let rec loop ds (xs:D<'a>) =
            match ds,xs with
            | [],_                        -> xs
            | d::[],T ds when d=ds.Length -> T ds
            | d::[],F ds when d=ds.Length -> F ds
            | d::rest,T ds -> loop rest (ds |> Array.chunkBySize d |> Array.map T |> T)
            | d::rest,F ds -> loop rest (ds |> Array.chunkBySize d |> Array.map F |> T)
        loop rdims (F ts)
