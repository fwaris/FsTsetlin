#load "packages.fsx"
open FsTsetlin
open TorchSharp

let numClasses = 3
let clausesPerClss = 5

let cl1 = torch.ones(2L * int64 numClasses * int64 clausesPerClss)
Utils.tensorData<float32> cl1

let cl2 = cl1.reshape(2L * int64 numClasses, -1L)

let cl3 = cl2.sum(1L)
Utils.tensorData<float32> cl3

let i = cl3.argmax()
i.ToInt64()

//clauses for i are treated as y = 1
//find random class (not i) and treat it as y = 0

(*
c0 - 0
c0 + 0
c0 - 1
c0 + 1
...
c0 - 4
c0 + 4

c1 - 0
c1 + 0
c2 - 0
c2 + 0
...
*)





