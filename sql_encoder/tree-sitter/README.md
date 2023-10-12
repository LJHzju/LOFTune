需要先编译得到sql.so，才能正常运行

### 环境准备
需要事先安装node，gcc，g++，其中glibc版本>=2.29

### 编译流程
1. 在本目录下，从github下载tree-sitter对应平台的二进制包（如tree-sitter-linux-arm64.gz）：https://github.com/tree-sitter/tree-sitter/releases
2. 运行以下命令：
```
gzip -dc ./tree-sitter-linux-arm64.gz > tree-sitter  # 解压
chmod 777 ./tree-sitter  # 添加权限
./tree-sitter generate  # 生成源码
mv ./scanner.cc ./src/scanner.cc
./tree-sitter parse test.sql  # 其实可以使用任意一个.sql文件
```
3. 最后可以在~/.cache/tree-sitter/lib目录下看到sql.so，拷贝到本目录即可
