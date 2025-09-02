import torch, time, json
dev='cuda'
res={}
res["torch"]=torch.__version__
res["cuda"]=torch.version.cuda
res["is_available"]=torch.cuda.is_available()
if not torch.cuda.is_available():
    print(json.dumps(res,indent=2)); raise SystemExit
torch.cuda.synchronize()
M=4096
def bench():
    x=torch.randn(M,M,device=dev,dtype=torch.float16)
    y=torch.randn(M,M,device=dev,dtype=torch.float16)
    torch.cuda.synchronize(); t0=time.time()
    z=x@y
    torch.cuda.synchronize()
    dt=time.time()-t0
    tflops=(2*M**3)/dt/1e12
    return dt*1000,tflops,float(z.mean())
for _ in range(3): bench()  # warm-up
times=[]; tflops=[]
for _ in range(5):
    ms,tf,mean=bench()
    times.append(ms); tflops.append(tf)
res["matmul_4k_ms_list"]= [round(v,2) for v in times]
res["matmul_4k_ms_avg"]= round(sum(times)/len(times),2)
res["matmul_fp16_TFLOPS_avg"]= round(sum(tflops)/len(tflops),2)
print(json.dumps(res, indent=2))
