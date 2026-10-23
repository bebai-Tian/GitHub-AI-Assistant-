"""最小例子：Depends 到底是什么。用一个假"数据库"（其实是个字符串），

把注意力集中在 Depends 的行为上，而不是数据库上。

用 FastAPI 的 TestClient 跑，不需要启动服务器。
"""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

app = FastAPI()
LOG: list[str] = []


# ---------------------------------------------------------------------------
# 第一步：一个"提供资源"的函数。就把它想成"发一个数据库连接"。
# ---------------------------------------------------------------------------
def get_fake_db():
    LOG.append("  [get_fake_db] 开始：造一个'连接'")
    db = "连接#42"                       # 假装这是个数据库会话
    try:
        yield db                         # ← 把资源交出去，然后【停住】
    finally:
        LOG.append("  [get_fake_db] 结束：归还'连接'")


# ---------------------------------------------------------------------------
# 写法 A：不用 Depends。接口函数自己造资源。
# ---------------------------------------------------------------------------
@app.get("/without-depends")
def without_depends():
    LOG.append("  [接口 A] 开始干活")
    db = "连接#42"                       # 自己造
    try:
        return {"值": db, "说明": "我自己造的连接"}
    finally:
        LOG.append("  [接口 A] 自己归还")


# ---------------------------------------------------------------------------
# 写法 B：用 Depends。接口函数不造资源，只声明"我需要一个"。
# ---------------------------------------------------------------------------
@app.get("/with-depends")
def with_depends(db: str = Depends(get_fake_db)):     # ★ 关键就这一行
    LOG.append(f"  [接口 B] 开始干活，手上已经有：{db}")
    return {"值": db, "说明": "框架给我的连接"}


print("=" * 74)
print("写法 A：自己造资源")
print("=" * 74)
LOG.clear()
client = TestClient(app)
print("调用前 LOG 是空的:", LOG == [])
resp = client.get("/without-depends")
print("响应:", resp.json())
print("过程记录：")
for line in LOG:
    print(line)

print()
print("=" * 74)
print("写法 B：用 Depends —— 注意 LOG 里多了谁")
print("=" * 74)
LOG.clear()
print("调用前 LOG 是空的:", LOG == [])
resp = client.get("/with-depends")
print("响应:", resp.json())
print("过程记录：")
for line in LOG:
    print(line)

print()
print("=" * 74)
print("对比结论")
print("=" * 74)
print("写法 A 的过程：接口 A 开始 → 接口 A 自己归还")
print("写法 B 的过程：get_fake_db 开始 → 接口 B 干活 → get_fake_db 归还")
print()
print("差别就是：B 里那三行'造资源 / 归还资源'你【一行都不用写】。")
print("你只声明了 'db: str = Depends(get_fake_db)'，")
print("FastAPI 就替你在【进入函数前】调一次 get_fake_db()、")
print("把 yield 出来的值塞进 db，再在【函数返回后】把剩下的部分跑完（归还）。")

print()
print("=" * 74)
print("再验证一件事：Depends 是不是只是'调用一个函数'？")
print("=" * 74)
print("其实你也可以手动调 —— 只是很笨拙：")
gen = get_fake_db()                  # 注意：函数体【没有】立刻执行
print("  调用 get_fake_db() 后，LOG 还是空的:", LOG == [])
value = next(gen)                    # 推一下，才跑到 yield
print("  next() 之后拿到:", value)
print("  此时 LOG:", LOG[-1])
next(gen, None)                      # 再推一下，触发 finally
print("  再 next() 之后 LOG:", LOG[-1])
print()
print("→ Depends 做的就是这件事，只不过由框架自动完成。")
