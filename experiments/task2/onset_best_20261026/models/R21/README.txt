冻结模型：Old410-R21
保留用途：old410，历史代理IoU=0.754463924。
两个人口不能横比；二月已反复开发后查看，不是独立盲测；官方S_queue=null。
权重、路由、阈值、原始特征代码按字节复制，完整多轮依赖位于payload/outputs。
模型与代码可独立于原实验目录使用；Python依赖环境及公开原始数据另行提供。本保存动作没有安装或更改环境。
完整性：python -X utf8 verify.py
推理：python -X utf8 predict.py --data-root <kaggle_public目录> --split validation --output <结果.csv>
可用--split private；CPU最多4线程。缓存仅写work目录，不修改冻结权重。
Task1不作为Task2输入；共同冻结Task1另在总归档Task1-frozen目录备份。
后续训练必须使用新目录，不得修改本包。
