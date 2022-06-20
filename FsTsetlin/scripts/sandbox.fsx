#load "packages.fsx"
open FsTsetlin
open TorchSharp
open System
let device = if torch.cuda.is_available() then torch.CUDA else torch.CPU

let x = torch.tensor(6)
let t2 = torch.tensor(2)
let m1 = t2.pow(torch.arange(8, dtype=x.dtype))
m1.data<int>().ToArray()
torch.bit


let cls = torch.tensor([|for i in 1 .. 2*6 -> i |], dimensions=[|2L;6L|])
cls.Handle
cls.ElementSize
cls.size()
let T_cls = Utils.tensorData<int> cls
let inp = 
    [
        [for i in 0 .. 5 -> 0]
        [for i in 0 .. 5 -> 1]
        [for i in 0 .. 5 -> 3]
    ]
    |> Seq.collect (fun x -> x)
    |> Seq.toArray
let tinp = torch.tensor(inp,dimensions = [|3L; 6L|])
let T_tinp = Utils.tensorData<int> tinp
let cls2 = cls.reshape(1L,-1L)
Utils.tensorData<int> cls2
let cls3 = cls2.broadcast_to(tinp.shape)

let cfg =
    { Config.Default with
        s           = 3.0f
        T           = 10.0f
        TAStates    = 1000
        dtype       = torch.int16
        Device      = torch.CPU
        InputSize   = 100
        ClausesPerClass = 100 //total = 10 * 4000 = 40000
        Classes         = 10
    }

let tm = TM.create cfg

let payout = tm.TMState.PayoutMatrix

let dt = torch.int64

let i1 = torch.tensor([|0|],dtype=dt, dimensions = [|1L;1L|])
let i2 = torch.tensor([|0|],dtype=dt, dimensions = [|1L;1L|])
let i3 = torch.tensor([|0|],dtype=dt, dimensions = [|1L;1L|])
let i4 = torch.tensor([|0|],dtype=dt, dimensions = [|1L;1L|])
let i5 = torch.tensor([|0|],dtype=dt, dimensions = [|1L;1L|])

let vs = payout.[[|i1; i2; i3; i4; i5|]]

let asx  = System.Enum.GetValues<torch.ScalarType>()
let asx2 = torch.ScalarType.Int32.ToString()
let b:torch.ScalarType = System.Enum.Parse(asx2)
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

printfn "%B" (uint (1<<<31))

printfn "%B" ((~~~0) - 1)

let la_chunks = (2*784-1) / 32 + 1

let CLAUSES = 2000u
let INT_SIZE= 32u
let FEATURES = 784u
let CLAUSE_CHUNKS = ((CLAUSES - 1u)/INT_SIZE + 1u)

let filter : uint32 =
    if ((FEATURES*2u) % 32u > 0u) then
        (~~~(0xffffffffu <<< (int ((FEATURES*2u) % INT_SIZE)))) 
    else
        0xffffffffu

printfn "%B" (0x55555555)
printfn "%B" (0xaaaaaaaa)
let t1 = torch.tensor([|1|])


//clauses 0x55555555 i are treated as y = 1
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





