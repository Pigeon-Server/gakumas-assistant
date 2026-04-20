# Assets Layout

独立包期望的资源目录结构：

```text
assets/
  gakumasu-diff/
    *.yaml
  GakumasTranslationData/
    local-files/
      masterTrans/
        *.json
```

可选做法：

- 直接把原项目中的资源目录复制到这里
- 或通过环境变量把代码指向任意外部资源目录
