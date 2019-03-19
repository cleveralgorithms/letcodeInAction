# letcodeInAction
力扣题目解法、思路、实现
## 推荐刷题利器
vscode + leetcode 插件：https://github.com/jdneo/vscode-leetcode

## 规范

### 成员代码提交规范

1. 每个成员可根据自己的`名称`或者`昵称`新建自己的文件夹且必须在根目录下新建
  - 例如：

```
reverse@xiaomiwujiecao  // 推荐使用这种方式新建文件夹
```

2. 每成员在自己的文件夹下更新自己的代码，每个 `demo` 必须包含自己的 `README.md`
    - 例如

  ```
  └── reverse@xiaomiwujiecao
└── demo1
    ├── README.md
    └── demo.py

  ```

    - `README.md` 中可以描述文件的目录 ，方便其他成员查看


3. 新建的文件类型可根据自己的主要开发语言更新，切记不能修改公共文件、，例如本文件，公共文件只能管理员修改

4. 每个成员可以通过 `fork` 的方式 `fork` 到自己的项目下，每次更新需要更新并添加提交， 最后在 `github` 上 `pullRequest` 到主分支 ，由管理员审核通过之后进行合并分支

### `fork` 的分支下如何更新远程分支到本地?

#### 方法

请阅读相关问题 ：  [点我](https://github.com/cleveralgorithms/letcodeInAction/issues/4)

### 记住流程

1. fork 本分支（主分支）
2. 添加成员自己的文件夹，所有改动仅在自己的文件夹下更改
3. 提交到自己的远程分值之前 记得 `git pull upstream master`
4. 添加你的文件 并提交到自己的远程分支
5. 需要 `pull Request` 直接在 `github` 的自己的 `master` 分支 进行操作
