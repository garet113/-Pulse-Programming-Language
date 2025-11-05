#!/usr/bin/env python3
# Tiny Pulse v0 interpreter (reactive graph + time scheduler)
# NOTE: Prototype for demo; not production-safe.

import math, time, re, threading, queue, sys
from collections import defaultdict, deque

# ---------- Utilities ----------
def Hz(n): return 1.0/float(n)
def s(n): return float(n)
def ms(n): return float(n)/1000.0
def now(): return time.time()

# ---------- Signal graph ----------
class Signal:
    __slots__ = ("name","value","prev","deps","users","is_const","expr","env")
    def __init__(self, name, value=None, is_const=False):
        self.name=name; self.value=value; self.prev=None
        self.deps=set(); self.users=set()
        self.is_const=is_const
        self.expr=None; self.env=None

    def set_expr(self, expr, env, deps):
        self.expr=expr; self.env=env; self.deps=set(deps)
        for d in deps: d.users.add(self)

    def eval(self):
        if self.is_const or self.expr is None: return
        self.prev = self.value
        self.value = self.expr(self.env)

    def __repr__(self): return f"<Sig {self.name}={self.value}>"

class Env:
    def __init__(self):
        self.signals={}
        self.events=defaultdict(list)  # name -> list of handlers
        self.scheduler=Scheduler(self)
        self.std = StdLib(self)

    def get_sig(self,name,create=True,is_const=False):
        if name in self.signals: return self.signals[name]
        if not create: return None
        s=Signal(name,is_const=is_const); self.signals[name]=s; return s

    def set_const(self,name,val):
        s=self.get_sig(name,True,True); s.value=val; return s

    def set_let(self,name,expr,dep_names):
        deps=[self.get_sig(d,True) for d in dep_names]
        s=self.get_sig(name,True,False)
        s.set_expr(expr,self,deps)
        s.eval()
        return s

    def reeval_from(self, sig):
        # simple topo via BFS
        q=deque([sig])
        visited=set()
        while q:
            cur=q.popleft()
            if cur in visited: continue
            visited.add(cur)
            cur.eval()
            for u in cur.users: q.append(u)

    def emit(self, name, payload=None):
        for handler in list(self.events[name]):
            handler(payload)

class Scheduler:
    def __init__(self, env):
        self.env=env
        self.tasks=[]
        self.q=queue.PriorityQueue()
        self.running=False
    def after(self, delay, fn):
        self.q.put((now()+delay, fn))
    def every(self, period, fn):
        def wrap():
            fn()
            self.after(period, wrap)
        self.after(period, wrap)
    def rate(self, hz, fn):
        self.every(Hz(hz), fn)
    def run_forever(self):
        self.running=True
        while self.running:
            try:
                when, fn = self.q.get(timeout=0.05)
            except queue.Empty:
                continue
            dt = when - now()
            if dt>0:
                time.sleep(dt)
            fn()
    def stop(self):
        self.running=False

# ---------- Stdlib ----------
class StdLib:
    def __init__(self, env): self.env=env
    # math
    def sin(self,x): return math.sin(float(x))
    def clamp(self,x,a,b): return max(a,min(b,x))
    # signals
    def smooth(self, sigval, tau):
        # simple exponential smoothing placeholder
        return 0.85*sigval + 0.15*float(tau)
    # io stubs
    sensors = defaultdict(lambda: 20.0)
    actuators = defaultdict(lambda: None)
    def sensor(self, name): return self.sensors[name]
    def actuator(self, name, val=None):
        if val is None: return self.actuators[name]
        self.actuators[name]=val; return val
    # util
    def print(self,*args): print(*args)

# ---------- Lexer (ultra-light) ----------
TOKEN=re.compile(r"""
    \s*(?:
      (?P<NUM>\d+(?:\.\d+)?) |
      (?P<STR>"[^"]*") |
      (?P<KW>@rate|@after|@every|@when|const|let|emit|fn|true|false) |
      (?P<ID>[A-Za-z_][A-Za-z0-9_]*) |
      (?P<SYM>==|!=|<=|>=|->|[-+/*%(){}.,:|=<>]) |
      (?P<NL>
)
    )
""", re.VERBOSE)

def tokenize(s):
    pos=0; tokens=[]
    while pos<len(s):
        m=TOKEN.match(s,pos)
        if not m: raise SyntaxError(f"Bad token near: {s[pos:pos+20]!r}")
        pos=m.end()
        kind=m.lastgroup; val=m.group(kind)
        if kind=="NL": continue
        tokens.append((kind,val))
    tokens.append(("EOF",""))
    return tokens

# ---------- Parser (subset) ----------
class Parser:
    def __init__(self, src): self.toks=tokenize(src); self.i=0
    def peek(self): return self.toks[self.i]
    def eat(self,kind=None,val=None):
        t=self.toks[self.i]; self.i+=1
        if kind and t[0]!=kind: raise SyntaxError(f"Expected {kind}, got {t}")
        if val and t[1]!=val: raise SyntaxError(f"Expected {val}, got {t}")
        return t

    def parse(self):
        items=[]
        while self.peek()[0]!="EOF":
            if self._is("@rate"): items.append(self.rate_block())
            elif self._is("@after"): items.append(self.after_block())
            elif self._is("@every"): items.append(self.every_block())
            elif self._is("@when"): items.append(self.when_block())
            elif self._is("KW","const"): items.append(self.const_decl())
            elif self._is("KW","let"): items.append(self.let_decl())
            else: items.append(self.stmt())
        return items

    def _is(self, kind, val=None):
        t=self.peek()
        return t[0]==kind and (val is None or t[1]==val)

    # decls
    def const_decl(self):
        self.eat("KW","const"); name=self.eat("ID")[1]; self.eat("SYM","=")
        expr=self.expr(); return ("const", name, expr)
    def let_decl(self):
        self.eat("KW","let"); name=self.eat("ID")[1]; self.eat("SYM","=")
        expr=self.expr(); return ("let", name, expr)

    # timed
    def rate_block(self):
        self.eat("KW","@rate"); hz=float(self.eat("NUM")[1]); self.eat("SYM","Hz"); self.eat("SYM",":")
        body=self.block(); return ("rate", hz, body)
    def after_block(self):
        self.eat("KW","@after"); dur=self.duration(); self.eat("SYM",":")
        body=self.block(); return ("after", dur, body)
    def every_block(self):
        self.eat("KW","@every"); dur=self.duration(); self.eat("SYM",":")
        body=self.block(); return ("every", dur, body)
    def when_block(self):
        self.eat("KW","@when"); ev=self.eat("ID")[1]
        if self._is("SYM","("):
            self.eat("SYM","("); args=[]; 
            if not self._is("SYM",")"):
                args.append(self.eat("ID")[1])  # name-only
            self.eat("SYM",")")
        self.eat("SYM",":"); body=self.block()
        return ("when", ev, body)

    def duration(self):
        num=float(self.eat("NUM")[1])
        unit=self.eat("ID")[1]
        if unit=="ms": return ms(num)
        if unit=="s": return s(num)
        raise SyntaxError("duration must be ms or s")

    def block(self):
        self.eat("SYM","{")
        items=[]
        while not self._is("SYM","}"):
            items.append(self.stmt())
        self.eat("SYM","}")
        return items

    # stmts
    def stmt(self):
        if self._is("KW","emit"):
            self.eat("KW","emit"); ev=self.eat("ID")[1]
            self.eat("SYM","("); payload=None
            if not self._is("SYM",")"):
                payload=self.expr()
            self.eat("SYM",")")
            return ("emit",ev,payload)
        # assignment or expr
        if self.peek()[0]=="ID" and self.toks[self.i+1]==("SYM","="):
            name=self.eat("ID")[1]; self.eat("SYM","="); e=self.expr(); return ("assign",name,e)
        e=self.expr(); return ("expr", e)

    # expressions (Pratt-ish)
    def expr(self): return self.logic_or()
    def logic_or(self):
        node=self.logic_and()
        while self._is("SYM","|"):
            self.eat("SYM","|"); self.eat("SYM","|"); right=self.logic_and()
            node=("bin","or",node,right)
        return node
    def logic_and(self):
        node=self.equality()
        while self._is("SYM","&"):
            self.eat("SYM","&"); self.eat("SYM","&"); right=self.equality()
            node=("bin","and",node,right)
        return node
    def equality(self):
        node=self.compare()
        while self._is("SYM","==") or self._is("SYM","!="):
            op=self.eat("SYM")[1]; right=self.compare()
            node=("bin",op,node,right)
        return node
    def compare(self):
        node=self.term()
        while self._is("SYM","<") or self._is("SYM",">") or self._is("SYM","<=") or self._is("SYM",">="):
            op=self.eat("SYM")[1]; right=self.term()
            node=("bin",op,node,right)
        return node
    def term(self):
        node=self.factor()
        while self._is("SYM","+") or self._is("SYM","-"):
            op=self.eat("SYM")[1]; right=self.factor()
            node=("bin",op,node,right)
        return node
    def factor(self):
        node=self.unary()
        while self._is("SYM","*") or self._is("SYM","/") or self._is("SYM","%"):
            op=self.eat("SYM")[1]; right=self.unary()
            node=("bin",op,node,right)
        return node
    def unary(self):
        if self._is("SYM","-"): self.eat("SYM","-"); return ("neg", self.unary())
        if self._is("SYM","~"): self.eat("SYM","~"); return ("prev", self.unary())
        return self.call()
    def call(self):
        node=self.primary()
        while self._is("SYM","(") or (self._is("SYM",".") and True):
            if self._is("SYM","("):
                self.eat("SYM","("); args=[]
                if not self._is("SYM",")"):
                    args.append(self.expr())
                    while self._is("SYM",","):
                        self.eat("SYM",","); args.append(self.expr())
                self.eat("SYM",")")
                node=("call",node,args)
            elif self._is("SYM","."):
                self.eat("SYM","."); attr=self.eat("ID")[1]
                node=("attr",node,attr)
        return node
    def primary(self):
        kind,val=self.peek()
        if kind=="NUM": self.eat("NUM"); return ("num", float(val))
        if kind=="STR": self.eat("STR"); return ("str", val[1:-1])
        if kind=="KW" and val in ("true","false"):
            self.eat("KW"); return ("num", 1.0 if val=="true" else 0.0)
        if kind=="ID":
            self.eat("ID"); return ("id", val)
        if kind=="SYM" and val=="(":
            self.eat("SYM","("); e=self.expr(); self.eat("SYM",")"); return e
        raise SyntaxError(f"Unexpected token {self.peek()}")

# ---------- Evaluator ----------
class VM:
    def __init__(self, env): self.env=env

    def eval_expr(self, node):
        tag=node[0]
        if tag=="num": return node[1]
        if tag=="str": return node[1]
        if tag=="id":
            name=node[1]
            if name in BUILTINS: return BUILTINS[name]
            s=self.env.get_sig(name,False)
            if s is None or s.value is None: raise NameError(f"Unknown signal {name}")
            return s.value
        if tag=="prev":
            val=self.eval_expr(node[1])
            if node[1][0]=="id":
                s=self.env.get_sig(node[1][1],False)
                return s.prev if s else None
            return val
        if tag=="neg": return -self.eval_expr(node[1])
        if tag=="bin":
            op,nodeL,nodeR=node[1],node[2],node[3]
            L=self.eval_expr(nodeL); R=self.eval_expr(nodeR)
            return {
                "+": L+R, "-": L-R, "*": L*R, "/": L/R, "%": L%R,
                "==": 1.0 if L==R else 0.0,
                "!=": 1.0 if L!=R else 0.0,
                "<": 1.0 if L<R else 0.0,
                ">": 1.0 if L>R else 0.0,
                "<=": 1.0 if L<=R else 0.0,
                ">=": 1.0 if L>=R else 0.0,
            }[op]
        if tag=="call":
            callee=self.eval_expr(node[1])
            args=[self.eval_expr(a) for a in node[2]]
            return callee(*args)
        if tag=="attr":
            obj=self.eval_expr(node[1]); attr=node[2]
            return getattr(obj, attr) if hasattr(obj, attr) else None
        raise RuntimeError(f"unknown expr tag {tag}")

    def compile_lambda(self, expr_node):
        def fn(env):
            return self.eval_expr(expr_node)
        return fn

    def deps_of(self, node):
        names=set()
        def walk(n):
            t=n[0]
            if t=="id": names.add(n[1])
            for x in n[1:]:
                if isinstance(x,tuple): walk(x)
                elif isinstance(x,list):
                    for y in x:
                        if isinstance(y,tuple): walk(y)
        walk(node)
        return [n for n in names if n not in BUILTINS]

    def run(self, items):
        for it in items:
            kind=it[0]
            if kind=="const":
                _,name,expr=it
                val=self.eval_expr(expr)
                self.env.set_const(name,val)
            elif kind=="let":
                _,name,expr=it
                deps=self.deps_of(expr)
                fn=self.compile_lambda(expr)
                self.env.set_let(name,fn,deps)
            elif kind=="assign":
                _,name,expr=it
                s=self.env.get_sig(name,True)
                s.prev=s.value
                s.value=self.eval_expr(expr)
                self.env.reeval_from(s)
            elif kind=="expr":
                self.eval_expr(it[1])
            elif kind=="emit":
                _,ev,payload=it
                val=self.eval_expr(payload) if payload is not None else None
                self.env.emit(ev,val)
            elif kind in ("rate","every","after","when"):
                self.install_timed(it)
            else:
                raise RuntimeError(f"unknown item {it}")

    def install_timed(self, it):
        kind=it[0]
        if kind=="rate":
            _,hz,body=it
            def task():
                self.run(body)
            self.env.scheduler.rate(hz, task)
        elif kind=="every":
            _,dur,body=it
            def task():
                self.run(body)
            self.env.scheduler.every(dur, task)
        elif kind=="after":
            _,dur,body=it
            def task():
                self.run(body)
            self.env.scheduler.after(dur, task)
        elif kind=="when":
            _,ev,body=it
            def handler(payload):
                if any((tok[0]=="ID" and tok[1]=="payload") for tok in tokenize(str(body))):
                    self.env.set_const("payload", payload)
                self.run(body)
            self.env.events[ev].append(handler)

# ---------- Builtins exposed to Pulse ----------
def _wrap_std(method):
    def f(*args): return method(*args)
    return f

def make_builtins(env):
    std=env.std
    return {
        "sin": _wrap_std(std.sin),
        "clamp": _wrap_std(std.clamp),
        "smooth": _wrap_std(std.smooth),
        "sensor": _wrap_std(std.sensor),
        "actuator": _wrap_std(std.actuator),
        "print": _wrap_std(std.print),
        "now": now, "Hz": Hz, "s": s, "ms": ms,
    }

BUILTINS=None

# ---------- Runner ----------
def run_file(path):
    with open(path,"r",encoding="utf-8") as f: src=f.read()
    env=Env()
    global BUILTINS; BUILTINS=make_builtins(env)
    items=Parser(src).parse()
    vm=VM(env)
    vm.run(items)
    t=threading.Thread(target=env.scheduler.run_forever, daemon=True)
    t.start()
    try:
        while True: time.sleep(0.1)
    except KeyboardInterrupt:
        env.scheduler.stop()

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python3 pulse.py example.pulse"); sys.exit(1)
    run_file(sys.argv[1])
