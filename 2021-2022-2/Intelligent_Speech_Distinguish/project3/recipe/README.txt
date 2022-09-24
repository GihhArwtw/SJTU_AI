修改：
1）增加了一些程序执行位置的指示（echo "xxx Done";）
2）使用大语料训练语言模型（对local/aishell_train_lms.sh进行了修改，将第7行替换为text=data/data_aishell/transcript/train_large.txt）
3）只使用nnet3，不使用chain，因此注释了原run.sh中chain训练的代码。

使用方法：
./run.sh

预期的最终模型目录：
exp/nnet3/new_dnn_sp/final.mdl