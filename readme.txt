1、先安装pyaap和tqdm，直接pip它俩就行
2、data和label对应放在两个txt文件中，一个句子一行，句中每个字符空格分开
  data.txt：
      ...
      刘 艾 婷 的 女 票 是 黄 梓 琪 。
      ...
  label.txt:
      ...
      B-e I-e I-e O B-p B-p O B-e I-e I-e O
      ...
3、运行：python --input-path train_data.txt --label-path train_label.txt --test-input-path test_data.txt --test-label-path test_label.txt --savt-dir result

其中，train_data.txt train_label.txt test_data.txt test_label.txt 分别存放着你的训练数据标签和测试数据标签，根据存放文件名填入相应位置即可，--save-dir可以不用理会是干嘛的，每次随意给他一个文件名就可以。




