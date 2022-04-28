from typing import Optional, Union

from llvmlite import ir

import utils


INT   = ir.IntType(32)
FLOAT = ir.FloatType()
VOID  = ir.VoidType()


def get_llvm_prm_type(t):
    if t == 'INT' or t == 'CHAR':
        return INT
    elif t == 'FLOAT':
        return FLOAT
    else:
        return VOID


class BaseAST:
    def __init__(self, parent: Optional['BaseAST'] = None):
        self.parent = parent
        self.children = {}

    def set_parent(self, value: 'BaseAST'):
        self.parent = value

    def get_parent(self) -> Optional['BaseAST']:
        return self.parent

    def add_child(self, name, obj):
        obj.set_parent(self)
        self.children[name] = obj

    def get_children(self, name):
        return self.children.get(name) or self.parent and self.parent.get_children(name)

    def code_gen(self, module: ir.Module):
        pass


class CharLiteralAST(BaseAST):
    def __init__(self, value, parent=None):
        super().__init__(parent=parent)
        self.value = value

    def get_type(self):
        return 'CHAR'

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        return ir.Constant(INT, self.value)


class IntLiteralAST(BaseAST):
    def __init__(self, value, parent=None):
        super().__init__(parent=parent)
        self.value = value

    def get_type(self):
        return 'INT'

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        return ir.Constant(INT, self.value)


class FloatLiteralAST(BaseAST):
    def __init__(self, value, parent=None):
        super().__init__(parent=parent)
        self.value = value

    def get_type(self):
        return 'FLOAT'

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        return ir.Constant(FLOAT, self.value)


class VarDecAST(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.dim = 0
        self.name = ""
        self.type = None
        self.value = None
        self.ptr: Optional[Union[ir.Argument, ir.AllocaInstr, ir.GlobalVariable]] = None
        self.is_global = False

    def set_dim(self, value):
        self.dim = value
    
    def get_dim(self):
        return self.dim

    def set_name(self, value):
        self.name = value
    
    def get_name(self):
        return self.name

    def set_type(self, value):
        self.type = value

    def get_type(self):
        return self.type

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        t = get_llvm_prm_type(self.type)
        if self.get_parent().get_parent() is None:
            v = ir.GlobalVariable(module, t, self.name)
        else:
            v = builder.alloca(t, name=self.name)
        self.ptr = v
        print(v)
        return v


class VarDefAST(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.dim = -1
        self.value = None
        self.var_dec: Optional[VarDecAST] = None

    def set_dim(self, dim):
        self.dim = dim

    def get_dim(self):
        return self.dim

    def set_var_dec(self, obj):
        self.var_dec = obj

    def get_var_dec(self):
        return self.var_dec

    def set_value(self, value):
        self.value = value

    def get_type(self):
        return self.var_dec.type

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        if type(self.var_dec.ptr) is ir.Argument:
            return self.var_dec.ptr
        else:
            return builder.load(self.var_dec.ptr, name=self.var_dec.name)


class ProcedureCallAST(BaseAST):
    def __init__(self, proc, args, parent=None):
        super().__init__(parent=parent)
        self.proc_callee = proc
        self.args = args

    def set_parent(self, value):
        self.parent = value

    def is_valid(self):
        pass

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        args = []
        for a in self.args:
            args.append(a.code_gen(module, builder))
        print('proc')
        print(args)
        return builder.call(self.proc_callee.proc, args)


class FunctionCallAST(BaseAST):
    def __init__(self, func, args, parent=None):
        super().__init__(parent=parent)
        self.func_callee: FunctionDefAST = func
        self.args = args
        self.ret = None

    def set_parent(self, value):
        self.parent = value

    def set_ret_name(self, name):
        self.ret = name

    def get_type(self):
        return self.func_callee.type

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        args = []
        for a in self.args:
            args.append(a.code_gen(module, builder))
        print('func')
        print(args)
        return builder.call(self.func_callee.func, args, name="tmp")


class ReturnAst(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.value = None

    def set_value(self, value):
        self.value = value

    def code_gen(self, func, builder: Optional[ir.IRBuilder] = None):
        tmp = self.value.code_gen(func, builder)
        builder.ret(tmp)


class AssignmentAST(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.lval: Optional[VarDefAST] = None
        self.rval = None

    def set_lval(self, value):
        self.lval = value

    def set_rval(self, value):
        self.rval = value

    def is_valid(self):
        if self.lval.get_type() == self.rval.get_type():
            return True
        else:
            return False

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        rval_code = self.rval.code_gen(module, builder)
        builder.store(rval_code, self.lval.var_dec.ptr)
        return self.lval.var_dec.ptr


class BinaryAST(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.operator = None
        self.lhs: Optional[Union['BinaryAST', VarDefAST]] = None
        self.rhs: Optional[Union['BinaryAST', VarDefAST]] = None

    def set_lhs(self, value):
        self.lhs = value

    def set_rhs(self, value):
        self.rhs = value

    def set_op(self, value):
        self.operator = value

    def is_valid(self):
        if utils.is_operator(self.operator) and (self.lhs is not None) and (self.rhs is not None):
            return True
        else:
            return False

    def get_type(self):
        t1 = self.lhs.get_type()
        t2 = self.rhs.get_type()
        if t1 == t2 and utils.is_arithmetic_operator(self.operator):
            return t1
        else:
            return None

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        code_lhs = self.lhs.code_gen(module, builder)
        code_rhs = self.rhs.code_gen(module, builder)
        if code_lhs is None or code_rhs is None:
            return None
        if self.operator == 'AND':
            if self.lhs.get_type() == 'INT':
                return builder.and_(code_lhs, code_rhs, 'andtmp')
            elif self.lhs.get_type() == 'FLOAT':
                return None
        elif self.operator == 'OR':
            if self.lhs.get_type() == 'INT':
                return builder.or_(code_lhs, code_rhs, 'ortmp')
            elif self.lhs.get_type() == 'FLOAT':
                return None
        if self.operator == 'ADD':
            if self.lhs.get_type() == 'INT':
                return builder.add(code_lhs, code_rhs, 'addtmp')
            elif self.lhs.get_type() == 'FLOAT':
                return builder.fadd(code_lhs, code_rhs, 'addtmp')
        elif self.operator == 'SUB':
            if self.lhs.get_type() == 'INT':
                return builder.sub(code_lhs, code_rhs, 'subtmp')
            elif self.lhs.get_type() == 'FLOAT':
                return builder.fsub(code_lhs, code_rhs, 'subtmp')
        elif self.operator == 'DIV':
            if self.lhs.get_type() == 'INT':
                return builder.udiv(code_lhs, code_rhs, 'divtmp')
            elif self.lhs.get_type() == 'FLOAT':
                return builder.fdiv(code_lhs, code_rhs, 'divtmp')
        elif self.operator == 'MUL':
            if self.lhs.get_type() == 'INT':
                return builder.mul(code_lhs, code_rhs, 'multmp')
            elif self.lhs.get_type() == 'FLOAT':
                return builder.fmul(code_lhs, code_rhs, 'multmp')
        elif self.operator == 'LT':
            if self.lhs.get_type() == 'INT':
                return builder.icmp_signed('<', code_lhs, code_rhs, 'lttmp')
            elif self.lhs.get_type() == 'FLOAT':
                return builder.fcmp_ordered('<', code_lhs, code_rhs, 'lttmp')
        elif self.operator == 'LE':
            if self.lhs.get_type() == 'INT':
                return builder.icmp_signed('<=', code_lhs, code_rhs, 'letmp')
            elif self.lhs.get_type() == 'FLOAT':
                return builder.fcmp_ordered('<=', code_lhs, code_rhs, 'letmp')
        elif self.operator == 'GT':
            if self.lhs.get_type() == 'INT':
                return builder.icmp_signed('>', code_lhs, code_rhs, 'gttmp')
            elif self.lhs.get_type() == 'FLOAT':
                return builder.fcmp_ordered('>', code_lhs, code_rhs, 'gttmp')
        elif self.operator == 'GE':
            if self.lhs.get_type() == 'INT':
                return builder.icmp_signed('>=', code_lhs, code_rhs, 'getmp')
            elif self.lhs.get_type() == 'FLOAT':
                return builder.fcmp_ordered('>=', code_lhs, code_rhs, 'getmp')
        elif self.operator == 'EQ':
            if self.lhs.get_type() == 'INT':
                return builder.icmp_signed('==', code_lhs, code_rhs, 'eqtmp')
            elif self.lhs.get_type() == 'FLOAT':
                return builder.fcmp_ordered('==', code_lhs, code_rhs, 'eqtmp')
        elif self.operator == 'NE':
            if self.lhs.get_type() == 'INT':
                return builder.icmp_signed('!=', code_lhs, code_rhs, 'netmp')
            elif self.lhs.get_type() == 'FLOAT':
                return builder.fcmp_ordered('!=', code_lhs, code_rhs, 'netmp')
        else:
            return None


class CompoundExpression(BaseAST):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.order_operations = []

    def set_child(self, obj):
        obj.set_parent(self)
        self.order_operations.append(obj)

    def get_var_def(self, name):
        ops = self.order_operations.copy()
        ops.reverse()
        for o in ops:
            if isinstance(o, AssignmentAST) and o.lval.var_dec.name == name:
                return o.lval
        return None

    def code_gen(self, module: ir.Module, bb: Optional[ir.Block] = None):
        for op in self.order_operations:
            op.code_gen(module, bb)
        return module


class ExprIfAST(CompoundExpression):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.expression = None
        self.then_body = None
        self.else_body = None

    def set_expression(self, expr):
        self.expression = expr

    def set_then(self, obj):
        self.then_body = obj

    def set_else(self, obj):
        self.else_body = obj

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        expr = self.expression.code_gen(module, builder)

        func = builder.basic_block.function

        then_block = func.append_basic_block('then')
        else_block = func.append_basic_block('else')
        merge_block = func.append_basic_block('ifcond')

        builder.cbranch(expr, then_block, else_block)

        builder.position_at_end(then_block)
        then_value = self.then_body.code_gen(module, builder)
        builder.branch(merge_block)

        builder.position_at_end(else_block)
        else_value = self.else_body and self.else_body.code_gen(module, builder)
        builder.branch(merge_block)

        builder.position_at_end(merge_block)

        return merge_block


class ExprWhileAST(CompoundExpression):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.expression = None
        self.body = None

    def set_expression(self, expr):
        self.expression = expr

    def set_body(self, obj):
        self.body = obj

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        func = builder.basic_block.function

        expr_block = func.append_basic_block('expr')
        body_loop = func.append_basic_block('loop')
        after_block = func.append_basic_block('after')

        builder.branch(expr_block)

        builder.position_at_end(expr_block)
        expr = self.expression.code_gen(module, builder)
        builder.cbranch(expr, body_loop, after_block)

        expr_block = builder.basic_block

        builder.position_at_end(body_loop)
        body_code = self.body.code_gen(module, builder)
        builder.branch(expr_block)

        body_loop = builder.basic_block
        builder.position_at_end(after_block)
        after_block = builder.basic_block

        return after_block


class ExprDoWhileAST(CompoundExpression):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.expression = None
        self.body = None

    def set_expression(self, expr):
        self.expression = expr

    def set_body(self, obj):
        self.body = obj

    def code_gen(self, module: ir.Module, builder: Optional[ir.IRBuilder] = None):
        func = builder.basic_block.function

        body_loop = func.append_basic_block('loop')
        expr_block = func.append_basic_block('expr_block')
        before_loop = func.append_basic_block('before')

        builder.branch(body_loop)

        builder.position_at_end(body_loop)
        body_code = self.body.code_gen(module, builder)
        builder.branch(expr_block)

        body_loop = builder.basic_block

        builder.position_at_end(expr_block)
        expr = self.expression.code_gen(module, builder)
        builder.cbranch(expr, body_loop, before_loop)

        expr_block = builder.basic_block
        builder.position_at_end(before_loop)
        before_loop = builder.basic_block

        return before_loop


class ProcedureDefAST(CompoundExpression):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.name = ""
        self.args: list[VarDecAST] = []
        self.body = None
        self.proc: Optional[ir.Function] = None

    def set_name(self, value):
        self.name = value

    def set_body(self, obj):
        self.body = obj

    def add_arg(self, arg: VarDecAST):
        self.args.append(arg)
        self.add_child(arg.name, arg)

    def code_gen(self, module: ir.Module, bb: Optional[ir.Block] = None):
        args_type = [get_llvm_prm_type(arg.type) for arg in self.args]
        ty_func = ir.FunctionType(VOID, args_type)
        func = ir.Function(module, ty_func, self.name)

        self.proc = func

        for i in range(len(self.args)):
            func.args[i].name = self.args[i].name
            self.args[i].ptr = func.args[i]

        bb = func.append_basic_block("entry")
        builder = ir.IRBuilder(bb)

        for op in self.order_operations:
            op.code_gen(func, builder)
        if builder.block.terminator is None:
            builder.ret_void()


class FunctionDefAST(CompoundExpression):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.name = ""
        self.args: list[VarDecAST] = []
        self.return_value = None
        self.type = None
        self.body = None
        self.func: Optional[ir.Function] = None

    def set_name(self, value):
        self.name = value

    def set_body(self, obj):
        self.body = obj

    def set_type(self, t):
        self.type = t

    def add_arg(self, arg: VarDecAST):
        self.args.append(arg)
        self.add_child(arg.name, arg)

    def set_return_value(self, obj):
        self.return_value = obj

    def code_gen(self, module: ir.Module, bb: Optional[ir.Block] = None):
        ret_type = get_llvm_prm_type(self.type)

        args_type = [get_llvm_prm_type(arg.type) for arg in self.args]
        ty_func = ir.FunctionType(ret_type, args_type)
        func = ir.Function(module, ty_func, self.name)

        self.func = func

        for i in range(len(self.args)):
            func.args[i].name = self.args[i].name
            self.args[i].ptr = func.args[i]

        bb = func.append_basic_block("entry")
        builder = ir.IRBuilder(bb)

        for op in self.order_operations:
            op.code_gen(func, builder)
