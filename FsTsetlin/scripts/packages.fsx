#r "nuget: TorchSharp"
#r "nuget: Plotly.NET"

let LIBTORCH = 
    let path = System.Environment.GetEnvironmentVariable("LIBTORCH")
    if path <> null then path else @"D:\s\libtorch\lib\torch_cuda.dll"
System.Runtime.InteropServices.NativeLibrary.Load(LIBTORCH) |> ignore
let path = System.Environment.GetEnvironmentVariable("path")
let path' = $"{path};{LIBTORCH}"
System.Environment.SetEnvironmentVariable("path",path')

#load @"../FsTsetlin.fs"
