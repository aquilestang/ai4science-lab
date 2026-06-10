# 推送到 GitHub(一次性,需要你的账号)

我已完成 `git commit`(含全部代码 + 结果图),并打包成 `ai4science-lab.bundle`(完整历史)。
我(AI)**无法用你的 GitHub 身份推送**,需要你授权一次。

> ⚠️ 说明:这个文件夹在沙盒里是「只能新建、不能删除」的挂载,git 没法在里面维护 `.git` 元数据,
> 所以残留了一个**半成品 `.git/` 和 `_probe.txt`(无害)**。在你自己的 Mac 上(正常文件系统)按下面任一方式即可,先无视它们。

## 方式 0:用打包好的 bundle(保留我的 commit,最干净)
```bash
cd "<你的库路径>/Aq_doc"
git clone ai4science-lab/ai4science-lab.bundle ai4science-lab-repo
cd ai4science-lab-repo
# 在 github.com 新建空仓库(别勾 add README),把 USERNAME 换成你的:
git remote add origin https://github.com/USERNAME/ai4science-lab.git
git branch -M main && git push -u origin main
```

## 方式 A:就地初始化(在原文件夹,先清沙盒残留)
```bash
cd "<你的库路径>/Aq_doc/ai4science-lab"
rm -rf .git _probe.txt ai4science-lab.bundle   # 在 Mac 上可正常删除
git init && git add -A && git commit -m "init: ai4science-lab"
git remote add origin https://github.com/USERNAME/ai4science-lab.git
git branch -M main && git push -u origin main
```

## 方式 B:GitHub Desktop / VS Code
直接 "Add existing repository" → 选这个文件夹 → Publish。

## 让仓库更像「投名状」
- 在 GitHub 仓库 **About** 里写一句话定位 + 加 topics:`drug-discovery` `cheminformatics` `machine-learning` `rdkit`。
- 把这个仓库 **置顶(Pin)** 到你的 GitHub 主页。
- README 里的 parity 图会自动显示。

---
> 下一轮定时任务里,如果你已经把 GitHub 连接器接上了,我可以帮你把后续实验直接提交上去。
