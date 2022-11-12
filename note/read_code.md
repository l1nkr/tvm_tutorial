## 源码分析

### 框架图
![](../image/read_code/architecture_diagram.jpg)



### torch前端到tvm.IRModule
```
调用流程：relay.frontend.from_pytorch
1 tvm.IRModule(初始化容器，用于保存relay的信息)
2 Prelude(加载辅助函数)
  2.1 import_from_std(加载基础函数)
  2.2 tensor_array_ops.register(加载tensorarray相关函数)
3 PyTorchOpConverter(构建converter，用于算子解析)
4 create inputs && params
  4.1 _get_relay_input_vars(构建inputs)
  4.2 convert_params(构建params)
5 converter.convert_operators(转换算子)
6 set the IRModule
  6.1 analysis.free_vars(确定无依赖参数，例如inputs，params)
  6.2 tvm.relay.Function(用Function包装DAG计算过程)
  6.3 transform.RemoveUnusedFunctions(简单优化去除无用代码)
```
[参考](https://zhuanlan.zhihu.com/p/457039705)

### Python调用C++机制

[这篇文章写得很好，说得很清楚](https://blog.csdn.net/zx_ros/article/details/122931616)

[参考](https://zhuanlan.zhihu.com/p/363991566)
### autotvm执行过程
```python
调用流程：
1 relay.frontend.from_pytorch(解析过程，略)
2 autotvm.task.extract_from_program(任务提取)
  2.1 _lower(收集task)
    2.1.1 optimize(优化过程，略)
    2.1.2 GraphExecutorCodegen.codegen
      2.1.2.1(调用链)->select_implementation
        2.1.2.1.1 get_valid_implementations(通过FTVMStrategy找到实现)
        2.1.2.1.2 fcompute -> register_topi_compute
          2.1.2.1.2.1 _register_task_compute(函数添加到TASK_TABLE中)
          2.1.2.1.2.2 TaskExtractEnv.add_task(添加task)
  2.2 create(填充task信息)
    2.2.1 ApplyConfig(config_space)(设定默认config，用于记录解空间) && Task.func -> fcompute && fschedule(记录所有可能解)
    2.2.2 compute_flop(计算flop)
3 Tuner.tune(寻找解)
  3.1 create_measure_batch(创建评估函数)
    3.1.1 LocalRunner.set_task(初始化runner)
      3.1.1.1 Tracker.__init__(创建Tracker监听消息)
        3.1.1.1.1 _tracker_server -> TrackerServerHandler
        3.1.1.1.2 TrackerServerHandler.run(开始监听IO)
          3.1.1.1.2.1 _on_event->TCPEventHandler.on_message(事件处理)
            3.1.1.1.2.1.1 _init_conn(验证链接)
            3.1.1.1.2.1.2 call_handler -> ret_value(处理并回复)
      3.1.1.2 Server.__init__(创建Server监听消息)
        3.1.1.2.1 PopenWorker._start(创建tvm/exec/popen_worker.py子进程处理消息)
        3.1.1.2.2 _popen_start_rpc_server -> PopenRPCServerState -> _listen_loop(绑定监听线程)
          3.1.1.2.2.1 _listen_loop -> setup tracker(绑定tracker)
      3.1.1.3 check_remote(创建临时client并检查链接)
        3.1.1.3.1 connect_tracker(和tracker进行初始信息交互)
        3.1.1.3.2 TrackerSession.request(构建client并初始化session)
          3.1.1.3.2.1 sendjson && recvjson(获取tracker保存的key)
          3.1.1.3.2.2 _ffi_api.Connect交互(remote和server信息交互)
          3.1.1.3.2.3 RPCEndpoint::InitRemoteSession && ServerLoop -> HandleUntilReturnEvent -> HandleNextEvent (构建Remote链接和信息处理Loop)
    3.1.2 LocalBuilder.set_task(初始化builder)
  3.2 next_batch(取下一批数据) && measure_batch(评估得到结果)
    3.2.1 LocalBuilder.build(构建可执行的函数内核)
      3.2.1.1 LocalExecutor.submit -> _WrappedBuildFunc.__call__ -> _build_func_common -> tvm.driver.build(构建内核)
    3.2.2 LocalRunner.run(执行内核得到结果)
      3.2.2.1 module_loader(加载内核)
        3.2.2.1.1 request_remote(RPC链接)
        3.2.2.1.2 RPCSession.upload(使用tvm.rpc.server.upload保存lib)
          3.2.2.1.2.1.client RPCClientSession.GetFunction -> RPCEndpoint::SysCallRemote(获取函数)
          3.2.2.1.2.1.server HandleUntilReturnEvent -> HandleSyscall -> LocalSession::GetFunction(获取注册的函数)
          3.2.2.1.2.2.client RPCClientSession::CallFunc -> RPCEndpoint::CallFunc(调用函数)
          3.2.2.1.2.2.server HandleUntilReturnEvent -> HandleNormalCallFunc -> LocalSession::CallFunc(执行函数)
        3.2.2.1.3 RPCSession.load_module(使用tvm.rpc.server.load_module加载lib)
      3.2.2.2 mod.time_evaluator
        3.2.2.2.1 runtime.RPCTimeEvaluator(获取包裹的测试函数)
      3.2.2.3 random_fill
        3.2.2.3.1 get_function(获取函数)
        3.2.2.3.2 nd.array(构建原始数据)
          3.2.2.3.2.1 empty -> NDArray::Empty -> RPCDevAllocDataWithScope(开辟内存)
          3.2.2.3.2.2 copyfrom -> TVMArrayCopyFromBytes -> HandleCopyToRemote && HandleSyscallStreamSync(拷贝并同步)
        3.2.2.3.3 random_fill.__call__ -> RandomEngine::RandomFill(随即填充数据)
      3.2.2.4 dev.sync -> RPCSession::StreamSync -> HandleSyscallStreamSync(设备同步)
      3.2.2.5 time_f.__call__ -> WrapTimeEvaluator(执行测试，多次平均)
```


GridSearchTuner 也是一种调优器。继承关系：GridSearchTuner -> IndexBaseTuner -> Tuner
发现继承的子类似乎都只有获取数据的逻辑，在进行调优的时候，实际上用的都是Tuner类中的tune方法，也就是说调优的逻辑是相同的，只是获取数据的方式不同，这点算是比较好理解。

在看代码的时候，发现xgboost的cost model似乎并没有被使用。经过和老师讨论，认为应该是在使用遗传算法进行搜索时使用了cost model

[参考](https://zhuanlan.zhihu.com/p/457722423)

### autotvm.apply_history_best()

在使用relay.build之前，通常都会有这么一句
```python
with autotvm.apply_history_best(log_file):
    print("Compile...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    evaluate_performance(lib, data_shape)
```

log_file 是之前 tune 得到的数据，~~我猜测这里应该是训练成本模型的地方~~ 似乎不是，他只是从里面选了一些数据出来
ApplyHistoryBest 在 init 时会调用 self.load(records)

接下来进入 PassContext()
```python
def __init__(
    self,
    opt_level=2,
    required_pass=None,
    disabled_pass=None,
    instruments=None,
    config=None,
):
    ...
    self.__init_handle_by_constructor__(
        _ffi_transform_api.PassContext, opt_level, required, disabled, instruments, config
    )
```



### Build

tvm.build针对单一算子
relay.build针对整个模型进行编译

#### Build with Relay

高层逻辑：
1. 通过查询op注册表来查找op实现
2. 为op生成计算表达式和调度
3. 将op编译为目标代码

relay.build()函数会进入python/tvm/relay/build_module.py，首先判断有没有autotvm预先tune好记录，然后构造tophub_context
```python
if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
    tophub_context = autotvm.tophub.context(list(raw_targets))
else:
    tophub_context = autotvm.utils.EmptyContext()
```

在其内部构建了BuildModule之后，才跳转到BuildModule.build，然后返回BuildModule.__init__中的内容
```python
with tophub_context:
    bld_mod = BuildModule()
    graph_json, runtime_mod, params = bld_mod.build(
        some_args...
    )
```

```python
class BuildModule(object):
    """Build an IR module to run on TVM graph executor. This class is used
    to expose the `RelayBuildModule` APIs implemented in C++.
    """

    def __init__(self):
        # 将会进入c++层
        self.mod = _build_module._BuildModule()
        ...some init...
```

c++函数位于src/relay/backend/build_module.cc
```c++
runtime::Module RelayBuildCreate() {
  auto exec = make_object<RelayBuildModule>();
  return runtime::Module(exec);
}
TVM_REGISTER_GLOBAL("relay.build_module._BuildModule").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = RelayBuildCreate();
});
```

这里注册了RelayBuildCreate，RelayBuildCreate下面还使用了PackedFunc做了一层封装
```c++
else if (name == "build") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 8);
        this->Build(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
      });
```

调用的this->Build会再去调用BuildRelay
```c++
void Build(IRModule mod, const Array<Target>& raw_targets, const tvm::Target& target_host,
            const Executor& executor, const Runtime& runtime,
            const WorkspaceMemoryPools& workspace_memory_pools,
            const ConstantMemoryPools& constant_memory_pools, const String mod_name) {
    
    ...
    
    BuildRelay(std::move(mod), mod_name);
    }
```

下面的 BuildRelay 是核心模块，做了如下工作
1. 优化
2. 计算图生成
3. 后端代码生成
```c++
  /*!
   * \brief Compile a Relay IR module to runtime module.
   *
   * \param relay_module The Relay IR module.
   * \param params The parameters.
   */
  void BuildRelay(IRModule relay_module, const String& mod_name) {
    // Relay IRModule -> IRModule optimizations.
    IRModule module = WithAttrs(
        relay_module, {{tvm::attr::kExecutor, executor_}, {tvm::attr::kRuntime, runtime_}});
    // 优化
    relay_module = OptimizeImpl(std::move(module));

    // 获取更新的函数和新的 IRModule 来构建。
    // 与其重新创建 IRModule，不如查看它与传入的 IRModule 之间的区别，
    // 看看我们是否可以将 (IRModule, Function) 传递给代码生成器。
    Function func = Downcast<Function>(relay_module->Lookup("main"));
    IRModule func_module = WithAttrs(IRModule::FromExpr(func),
                                     {{tvm::attr::kExecutor, executor_},
                                      {tvm::attr::kRuntime, runtime_},
                                      {tvm::attr::kWorkspaceMemoryPools, workspace_memory_pools_},
                                      {tvm::attr::kConstantMemoryPools, constant_memory_pools_}});

    // Generate code for the updated function.
    // 计算图生成。判断是生成 GraphCodegen 还是 AOTCodegen
    executor_codegen_ = MakeExecutorCodegen(executor_->name);
    executor_codegen_->Init(nullptr, config_->primitive_targets);
    executor_codegen_->Codegen(func_module, func, mod_name);
    executor_codegen_->UpdateOutput(&ret_);
    ret_.params = executor_codegen_->GetParams();

    auto lowered_funcs = executor_codegen_->GetIRModule();

    // No need to build for external functions.
    Target ext_dev("ext_dev");
    if (lowered_funcs.find(ext_dev) != lowered_funcs.end()) {
      lowered_funcs.Set(ext_dev, IRModule());
    }

    const Target& host_target = config_->host_virtual_device->target;
    const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
    // When there is no lowered_funcs due to reasons such as optimization.
    if (lowered_funcs.size() == 0) {
      if (host_target->kind->name == "llvm") {
        CHECK(pf != nullptr) << "Unable to create empty module for llvm without llvm codegen.";
        // If we can decide the target is LLVM, we then create an empty LLVM module.
        ret_.mod = (*pf)(host_target->str(), "empty_module");
      } else {
        // If we cannot decide the target is LLVM, we create an empty CSourceModule.
        // The code content is initialized with ";" to prevent complaining
        // from CSourceModuleNode::SaveToFile.
        ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
      }
    } else {
      ret_.mod = tvm::TIRToRuntime(lowered_funcs, host_target);
    }

    auto ext_mods = executor_codegen_->GetExternalModules();
    ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, host_target,
                                                  runtime_, executor_,
                                                  executor_codegen_->GetExecutorCodegenMetadata());
    // Remove external params which were stored in metadata module.
    for (tvm::runtime::Module mod : ext_mods) {
      auto pf_var = mod.GetFunction("get_const_vars");
      if (pf_var != nullptr) {
        Array<String> variables = pf_var();
        for (size_t i = 0; i < variables.size(); i++) {
          auto it = ret_.params.find(variables[i].operator std::string());
          if (it != ret_.params.end()) {
            VLOG(1) << "constant '" << variables[i] << "' has been captured in external module";
            ret_.params.erase(it);
          }
        }
      }
    }
  }
```

1. 优化
优化Optimize，可以看到这里的优化主要是设备无关的优化，是graph-level的针对tensor运算的优化。不断往pass_seqs里面塞各种优化pass。
比如：
去除公共子表达式：EliminateCommonSubexpr
分支卷积优化：CombineParallelConv2D
常量传播优化：...
规范化：将一些特殊运算转换成等价的常规算子运算，主要就是bias_add 转换为 expand_dim + broadcast_add
layout 变换和常量传播
图融合优化
图融合优化。其优化内容几乎与 NNVM 一样，都是基于算子的 pattern (kElemWise, kBroadcast,kInjective, kCommReduce, kOutEWiseFusable, kOpaque)和融合规则 rule (kUknown, kFuseToMaster, kRealize)来运行融合算法的，可以参考一篇关于NNVM的文章，这里不再赘述。


```c++
// BuildRelay 中的 relay_module = OptimizeImpl(std::move(module));
IRModule OptimizeImpl(IRModule relay_module) {
    ICHECK(relay_module.defined()) << "The IRModule must be defined for the Relay compiler.";

    backend::BindParamsInModule(relay_module, params_);

    Array<Pass> pass_seqs =
        GetPassPrefix(/*is_homogenous=*/config_->primitive_targets.size() == 1, /*is_vm=*/false);
    transform::PassContext pass_ctx = PassContext::Current();

    if (config_->optional_homogeneous_target.defined()) {
      // This pass currently only supports the homogeneous case.
      pass_seqs.push_back(transform::SplitArgs(
          config_->optional_homogeneous_target->GetAttr<Integer>("max_function_args", -1)
              .value()
              .IntValue()));
    }

    // Always plan devices so the remaining passes don't need to distinguish homogeneous vs
    // hetrogenous execution.
    pass_seqs.push_back(transform::PlanDevices(config_));

    // Fuse the operations if it is needed.
    pass_seqs.push_back(transform::FuseOps());

    // Create a sequential pass and perform optimizations.
    transform::Pass seq = transform::Sequential(pass_seqs);
    if (config_->optional_homogeneous_target.defined()) {
      With<Target> tctx(config_->optional_homogeneous_target);
      relay_module = seq(relay_module);
    } else {
      relay_module = seq(relay_module);
    }

    // Do layout rewrite for auto-scheduler.
    if (backend::IsAutoSchedulerEnabled() && config_->optional_homogeneous_target.defined()) {
      Pass major_pass = transform::AutoSchedulerLayoutRewrite();
      bool enable_layout_rewrite_targets =
          config_->optional_homogeneous_target->kind->device_type == kDLCPU ||
          config_->optional_homogeneous_target->GetAttr<String>("device", "") == "mali";
      if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
        With<Target> tctx(config_->optional_homogeneous_target);
        relay_module = major_pass(relay_module);
        // Defuse ops to fold constants, then fuse them again
        relay_module = transform::DefuseOps()(relay_module);
        relay_module = transform::FoldConstant()(relay_module);
        relay_module = transform::FuseOps()(relay_module);
      }
    }
    // do layout rewrite for metaschedule
    if (backend::IsMetaScheduleEnabled() && config_->optional_homogeneous_target.defined()) {
      Pass major_pass = transform::MetaScheduleLayoutRewrite();
      bool enable_layout_rewrite_targets =
          config_->optional_homogeneous_target->kind->device_type == kDLCPU ||
          config_->optional_homogeneous_target->GetAttr<String>("device", "") == "mali";
      if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
        With<Target> tctx(config_->optional_homogeneous_target);
        relay_module = major_pass(relay_module);
        // Defuse ops to fold constants, then fuse them again
        relay_module = transform::DefuseOps()(relay_module);
        relay_module = transform::FoldConstant()(relay_module);
        relay_module = transform::FuseOps()(relay_module);
      }
    }

    relay_module = transform::InferType()(relay_module);

    // Inline the functions that have been lifted by the module scope.
    //
    // TODO(@zhiics) Note that we need to be careful about the subgraphs with
    // global function calls. We should make sure that these callees are also
    // inline functions. However, this should be very unlikely for accelerators
    // and vendor-provided libraries. So we don't handle for now.
    relay_module = transform::Inline()(relay_module);
    relay_module = transform::InferType()(relay_module);
    relay_module = transform::LabelOps()(relay_module);
    relay_module = transform::AnnotateMemoryScope(config_)(relay_module);

    ICHECK(relay_module.defined());

    return relay_module;
  }
```

2. 计算图生成

``BuildRelay`` 函数中有下面几行代码

```c++
    // Generate code for the updated function.
    // 计算图生成。判断是生成 GraphCodegen 还是 AOTCodegen
    executor_codegen_ = MakeExecutorCodegen(executor_->name);
    executor_codegen_->Init(nullptr, config_->primitive_targets);
    executor_codegen_->Codegen(func_module, func, mod_name);
```

``executor_codegen`` 的类型可能是 ``GraphCodegen`` 也可能是 ``AOTCodegen``

下面按照 GraphCodegen 的类型进行分析

调用 ``executor_codegen_->Codegen(func_module, func, mod_name);`` -> ``ExecutorCodegen::Codegen`` -> ``relay.build_module._GraphExecutorCodegen`` -> ``CreateGraphCodegenMod`` -> ``GraphExecutorCodegenModule`` 

下面给出 ``GraphExecutorCodegenModule`` 的代码

```c++
class GraphExecutorCodegenModule : public runtime::ModuleNode {
 public:
  GraphExecutorCodegenModule() {}
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "init") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2) << "The expected of arguments are: "
                                    << "runtime::Module mod and Array<Target> targets";
        void* mod = args[0];
        Array<Target> targets = args[1];
        codegen_ = std::make_shared<GraphExecutorCodegen>(reinterpret_cast<runtime::Module*>(mod),
                                                          std::move(targets));
      });
    } else if (name == "codegen") {
      // 在这里进行调用
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        IRModule mod = args[0];
        Function func = args[1];
        String mod_name = args[2];
        this->output_ = this->codegen_->Codegen(mod, func, mod_name);
      });
    } 
    ...
  }

  const char* type_key() const final { return "RelayGraphExecutorCodegenModule"; }

 private:
  std::shared_ptr<GraphExecutorCodegen> codegen_;
  LoweredOutput output_;
};
```

继续看 Codegen 的具体实现

遍历 relay::Function func，然后生成计算图。

内存分配：由函数relay.backend.GraphPlanMemory实现；src/relay/backend/graph_plan_memory.cc

VisitExpr对节点进行遍历并进行节点信息的记录。

LowerExternalfunctions完成ir节点到tir节点的转化以及schedule的优化。

[细节参考](https://zhuanlan.zhihu.com/p/339566528)
```c++
LoweredOutput Codegen(IRModule mod, relay::Function func, String mod_name) {
    mod_name_ = mod_name;
    memory_plan_ = GraphPlanMemory(func);
    backend::FunctionInfo func_info;
    if (memory_plan_.defined()) {
      func_info =
          relay::tec::UpdateMainWorkspaceSize(mod, config_, memory_plan_->expr_to_storage_info);
      mod = WithAttr(mod, "main_func_info", func_info);
    }
    IRModule lowered_mod = tec::LowerTE(mod_name_, config_, [this](BaseFunc func) {
      // We need to maintain the constant map for external
      // functions so we pass this processing function which
      // allows us to process each function as we lower it.
      if (func->GetAttr<String>(attr::kCompiler).defined()) {
        UpdateConstants(func, &params_);
      }
      tec::UpdateFunctionMetadata(func, this->function_metadata_);
    })(mod);

    Optional<backend::FunctionInfo> main_func_info =
        lowered_mod->GetAttr<backend::FunctionInfo>("main_func_info");

    function_metadata_.Set(runtime::symbol::tvm_module_main, main_func_info.value());
    Function lowered_main_func = Downcast<Function>(lowered_mod->Lookup("main"));

    // Now that we have lowered all operators to TIR code, we can proceed with compilation.
    // We need to unfortunately re-plan as the previous results have been invalidated by lowering
    // we will fix this in future refactors.
    memory_plan_ = GraphPlanMemory(lowered_main_func);
    // The graph planner also can not handle planning calls to global variables to we must remap
    // First we convert all the parameters into input nodes.
    for (auto param : lowered_main_func->params) {
      auto node_ptr = GraphInputNode::make_node_ptr(param->name_hint(), GraphAttrs());
      var_map_[param.get()] = AddNode(node_ptr, param);
    }
    heads_ = VisitExpr(lowered_main_func->body);
    std::ostringstream os;

    dmlc::JSONWriter writer(&os);
    GetJSON(&writer);
    LoweredOutput ret;
    ret.graph_json = os.str();

    // Collect any runtime modules generated by external codegen.
    ret.external_mods =
        lowered_mod->GetAttr<Array<runtime::Module>>(tvm::attr::kExternalMods).value_or({});

    // Collect any constants extracted by external codegen.
    ret.params = std::unordered_map<std::string, tvm::runtime::NDArray>();
    Map<String, runtime::NDArray> const_name_to_constant =
        lowered_mod->GetAttr<Map<String, runtime::NDArray>>(tvm::attr::kConstNameToConstant)
            .value_or({});
    for (const auto& kv : const_name_to_constant) {
      VLOG(1) << "constant '" << kv.first << "' contributed by external codegen";
      ICHECK(ret.params.emplace(kv.first, kv.second).second);
    }

    // Collect any constants extracted during lowering.
    for (const auto& kv : params_) {
      VLOG(1) << "constant '" << kv.first << "' contributed by TECompiler";
      ICHECK(ret.params.emplace(kv.first, kv.second).second);
    }

    ret.function_metadata = std::move(function_metadata_);

    // This is the point where we separate the functions in the module by target
    ret.lowered_funcs = tec::GetPerTargetModules(lowered_mod);
    ret.metadata =
        ExecutorCodegenMetadata({} /* inputs */, {} /* input_tensor_types */, {} /* outputs */,
                                {} /* output_tensor_types */, {} /* pools */, {} /* devices */,
                                runtime::kTvmExecutorGraph /* executor */, mod_name_ /* mod_name */,
                                "packed" /* interface_api */, Bool(false) /* unpacked_api */);
    return ret;
  }
```


3. 后端代码生成

Relay得到lower后的函数，将做后端代码生成，跳转到src/driver/driver_api.cc中的TIRToRuntime函数（注意这里重载了多种实现），然后跳转到核心build，这里的build函数支持异构编译，需要在inputs划分好不同硬件设施。
（其实不是很清楚怎么跳转到这个函数的）

```c++
runtime::Module TIRToRuntime(const Map<Target, IRModule>& inputs_arg,
                             const Target& target_host_arg) {
  auto pass_ctx = transform::PassContext::Current();

  std::vector<runtime::Module> device_modules;
  Map<Target, IRModule> inputs = inputs_arg;
  Target target_host = target_host_arg;

  // Fetch previous defined target host in targets
  CheckAndUpdateHostConsistency(&inputs, &target_host);

  if (!target_host.defined()) {
    for (const auto& it : inputs) {
      if (it.first->kind->device_type == kDLCPU || it.first->kind->device_type == kDLMicroDev) {
        target_host = it.first;
        break;
      }
    }
  }

  if (!target_host.defined()) {
    target_host = DefaultTargetHost(target_host);
  }

  // Update target host for all targets
  CheckAndUpdateHostConsistency(&inputs, &target_host);

  // Take the attrs from the first module so the eventual modules have them.
  // Ideally this would just be one unified module all the way through;
  IRModule first_module = (*inputs.begin()).second;
  IRModule mhost_all = IRModule(Map<GlobalVar, BaseFunc>(), {}, {}, {}, first_module->attrs);

  ICHECK(mhost_all.defined()) << "The host module must be defined";

  for (const auto& it : inputs) {
    if (it.second.defined()) {
      const Target& target = it.first;
      const IRModule& ir_module = it.second;
      auto pair = SplitMixedModule(ir_module, target, target_host);
      auto& host_mod = pair.first;
      auto& device_mod = pair.second;

      ICHECK(host_mod.defined()) << "The split host module must be defined";

      ICHECK(mhost_all.defined()) << "The host module must be defined";

      // We don't want library modules going back into host codegen
      // unless they're supposed to. Here if we overrode the target host
      // to allow lowering previously we check that it's meant to be placed
      // back into the host Module.
      bool overrides_host_target = target->kind->device_type == target_host->kind->device_type;
      bool non_host_target_kind = target->kind != target_host->kind;
      if (overrides_host_target && non_host_target_kind) {
        device_modules.push_back(codegen::Build(host_mod, it.first));
      } else {
        mhost_all->Update(host_mod);
      }

      if (device_mod->functions.size() != 0) {
        device_modules.push_back(codegen::Build(device_mod, it.first));
      }
    }
  }

  runtime::Module mhost = codegen::Build(mhost_all, target_host);
  for (const auto& it : device_modules) {
    if (it.operator->()) {
      mhost.Import(it);
    }
  }

  return mhost;
}
```

当中最最核心的则是mhost = codegen::Build，最后跳转过去就开始调用代码生成模块了（src/target/codegen.cc）。

```c++
runtime::Module Build(IRModule mod, Target target) {
  if (transform::PassContext::Current()
          ->GetConfig<Bool>("tir.disable_assert", Bool(false))
          .value()) {
    mod = tir::transform::SkipAssert()(mod);
  }

  auto target_attr_map = tvm::TargetKind::GetAttrMap<FTVMTIRToRuntime>("TIRToRuntime");
  if (target_attr_map.count(target->kind)) {
    return target_attr_map[target->kind](mod, target);
  }

  // the build function.
  std::string build_f_name = "target.build." + target->kind->name;
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  ICHECK(bf != nullptr) << build_f_name << " is not enabled";
  return (*bf)(mod, target);
}
```

[参考](https://zhuanlan.zhihu.com/p/381691430)

4. python

c++ 中的代码跑完了后，继续回到 python 的 build_module.py 中
```python
elif executor.name == "graph":
    executor_factory = _executor_factory.GraphExecutorFactoryModule(
        ir_mod,
        raw_targets,
        executor,
        graph_json,
        runtime_mod,
        mod_name,
        params,
        func_metadata,
    )
```

进入到 GraphExecutorFactoryModule 中
```python
def __init__(
    self,
    ir_mod,
    target,
    executor,
    graph_json_str,
    libmod,
    libmod_name,
    params,
    function_metadata,
):
    assert isinstance(graph_json_str, string_types)
    fcreate = get_global_func("tvm.graph_executor_factory.create")
    ...
    self.module = fcreate(graph_json_str, libmod, libmod_name, *args)
    ...
```

在 src/runtime/graph_executor/graph_executor_factory.cc 中注册这个函数
```python
TVM_REGISTER_GLOBAL("tvm.graph_executor_factory.create")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK_GE(args.num_args, 3) << "The expected number of arguments for "
                                     "graph_executor_factory.create needs at least 3, "
                                     "but it has "
                                  << args.num_args;
      // The argument order is graph_json, module, module_name, param0_name, param0_tensor,
      // [param1_name, param1_tensor], ...
      ICHECK_EQ((args.size() - 3) % 2, 0);
      std::unordered_map<std::string, tvm::runtime::NDArray> params;
      // 参数不止这些，利用map存多个参数
      for (size_t i = 3; i < static_cast<size_t>(args.size()); i += 2) {
        std::string name = args[i].operator String();
        params[name] = args[i + 1].operator tvm::runtime::NDArray();
      }
      auto exec = make_object<GraphExecutorFactory>(args[0], params, args[2]);
      exec->Import(args[1]);
      *rv = Module(exec);
    });
```

下面似乎是仅仅构造了出来，没有看到调用图优化的函数
```python
GraphExecutorFactory::GraphExecutorFactory(
    const std::string& graph_json,
    const std::unordered_map<std::string, tvm::runtime::NDArray>& params,
    const std::string& module_name) {
  graph_json_ = graph_json;
  params_ = params;
  module_name_ = module_name;
}
```

<!--  -->
在 build_module.py 中还会调用 bld_mod.build()函数
```python
bld_mod = BuildModule()
graph_json, runtime_mod, params = bld_mod.build(
    mod=ir_mod,
    target=raw_targets,
    params=params,
    executor=executor,
    runtime=runtime,
    workspace_memory_pools=workspace_memory_pools,
    constant_memory_pools=constant_memory_pools,
    mod_name=mod_name,
)
```

bld_mod.build()函数往下走会调用 self._build() 函数。
**在里面还有一个发现，居然有一个 meta_schedule 选项，不知道是不是之前说的那个 meta schedule**
```python
self._build(
    mod,
    target,
    target_host,
    executor,
    runtime,
    workspace_memory_pools,
    constant_memory_pools,
    mod_name,
)
```