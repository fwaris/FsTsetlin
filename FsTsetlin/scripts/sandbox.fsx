#load "packages.fsx"
open FsTsetlin
open TorchSharp

let numClasses = 3
let clausesPerClss = 5

let a = torch.tensor([|0.1f; -0.1f; 0.0f |])
let r = torch.rand_like(a)
let a_abs = a.abs()
let lt = a_abs.less_equal(a_abs)
let sel = a.where(lt,0.0f)

Utils.tensorData<float32> a
Utils.tensorData<float32> r
Utils.tensorData<float32> a_abs
Utils.tensorData<bool> lt
Utils.tensorData<float32> sel

let fb = torch.tensor([|1; -1; 0; 1;|])
let sts = torch.tensor([|true; false; true; false|])
let stsBin = torch.where(sts,1,-1)
let incrDecr = fb.mul(stsBin)
Utils.tensorData<int> incrDecr



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





