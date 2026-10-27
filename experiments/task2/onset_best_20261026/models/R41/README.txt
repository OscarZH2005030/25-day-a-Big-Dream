Phase367-R41：新367窗当前最佳参考；代理IoU=.790702988，Onset=.721513401，Ongoing=.859892575。旧410窗本模型=.750678877，低于另存R21的.754463924，不能合并或横比两人口最好成绩。
二月已反复查看，不是独立盲测；官方S_queue=null。Task1历史S_state=.942828633，原92文件hash不变，不作为Task2输入。
完整性：python -X utf8 verify.py
推理：python -X utf8 predict.py --data-root <kaggle_public> --split validation --output result.csv
可选--split private；公开原始数据和Python依赖环境需另行准备，CPU最多4线程；缓存写work，不改冻结模型。
