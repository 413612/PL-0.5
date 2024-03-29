from llvmlite import ir
from llvmlite import binding

import my_ast
import utils

from my_lexer import Lexer



def get_rpn(i, tokens):
    res, stack, error = [], [], ''
    while tokens[i].type not in ['SEMI', 'COLON']:
        if tokens[i].type == 'INT_LIT' or tokens[i].type == 'FLOAT_LIT' or tokens[i].type == 'CHAR_LIT':
            res.append(i)
            i += 1
            continue
        elif tokens[i].type == 'ID':
            res.append(i)
            if tokens[i + 1].type == 'LPAREN':
                i += 1
                while tokens[i].type != 'RPAREN':
                    i += 1
            else:
                i += 1
            continue
        elif utils.is_operator(tokens[i].type):
            if len(stack) == 0:
                stack.append(i)
            else:
                while tokens[stack[-1]].type != 'LPAREN' and utils.op_priority(tokens[i].type) >= utils.op_priority(tokens[stack[-1]].type):
                    res.append(stack.pop())
                    if len(stack) == 0:
                        break
                stack.append(i)
            i += 1
            continue
        elif tokens[i].type == 'LPAREN':
            stack.append(i)
            i += 1
            continue
        elif tokens[i].type == 'RPAREN':
            while tokens[stack[-1]].type != 'LPAREN':
                res.append(stack.pop())
                if len(stack) == 0:
                    break
            if len(stack) > 0:
                if tokens[stack[-1]].type == 'LPAREN':
                    stack.pop()
                else:
                    error = "В выражении неправильно расставлены скобки"
            i += 1
            continue
        else:
            break
    while len(stack) != 0:
        res.append(stack.pop())
    return i, res, error


def var_parse(i: int, tokens, parent):
    v = my_ast.VarDecAST()
    v.set_parent(parent)
    if tokens[i].type == 'VAR':
        i += 1
        if tokens[i].type == 'ID':
            obj = parent.get_children(tokens[i].value)
            if obj is not None:
                error = "Переменная с именем " + tokens[i].value + " существует."
                print(error)
                return None, i, error
            else:
                parent.add_child(tokens[i].value, v)
                v.set_name(tokens[i].value)
        else:
            error = "Ошибка объявления переменной. Не указано имя."
            print(error)
            return None, i, error
        i += 1
        if utils.is_type(tokens[i].type):
            v.set_type(tokens[i].type)
        else:
            error = "Ошибка объявления переменной. Некорректно указан тип."
            print(error)
            return None, i, error
        i += 1
        if tokens[i].type == 'SEMI':
            return v, i, ""
        else:
            error = "Ошибка. Нет точки с запятой."
            print(error)
            return None, i, error


def func_call_parse(i, tokens, parent):
    error = ""
    name = ""
    args = []
    if tokens[i].type == 'ID':
        name = tokens[i].value
    i += 1
    if tokens[i].type == 'LPAREN':
        i += 1
        while tokens[i].type != 'RPAREN':
            if utils.is_number(tokens[i].type):
                if tokens[i].type == 'INT_LIT':
                    numb = my_ast.IntLiteralAST(tokens[i].value)
                    args.append(numb)
                elif tokens[i].type == 'FLOAT_LIT':
                    numb = my_ast.FloatLiteralAST(tokens[i].value)
                    args.append(numb)
                elif tokens[i].type == 'CHAR_LIT':
                    char = my_ast.CharLiteralAST(tokens[i].value)
                    args.append(char)
            elif tokens[i].type == 'ID':
                obj = parent.get_children(tokens[i].value)
                if obj is None:
                    error = "Переменная с имененем " + tokens[i].value + " не объявлена."
                    print(error)
                    return None, i, error
                var_def_obj = my_ast.VarDefAST(parent)
                var_def_obj.set_var_dec(obj)
                args.append(var_def_obj)
            i += 1
    if name != "":
        obj = parent.get_children(name)
        if obj is not None:
            f = my_ast.FunctionCallAST(obj, args)
            f.set_parent(parent)
            return f, i, error
        else:
            error = "Не объявлена функция с именем " + name
            print(error)
            return None, i, error
    else:
        error = "Не корректное объявление функции"
        print(error)
        return None, i, error


def proc_call_parse(i, tokens, parent):
    error = ""
    name = ""
    args = []
    if tokens[i].type == 'ID':
        name = tokens[i].value
    i += 1
    while tokens[i].type != 'SEMI':
        if utils.is_number(tokens[i].type):
            if tokens[i].type == 'INT_LIT':
                numb = my_ast.IntLiteralAST(tokens[i].value)
                args.append(numb)
            elif tokens[i].type == 'FLOAT_LIT':
                numb = my_ast.FloatLiteralAST(tokens[i].value)
                args.append(numb)
            elif tokens[i].type == 'CHAR_LIT':
                char = my_ast.CharLiteralAST(tokens[i].value)
                args.append(char)
        elif tokens[i].type == 'ID':
            obj = parent.get_children(tokens[i].value)
            if obj is None:
                error = "Переменная с имененем " + tokens[i].value + " не объявлена."
                print(error)
                return None, i, error
            var_def_obj = my_ast.VarDefAST(parent)
            var_def_obj.set_var_dec(obj)
            args.append(var_def_obj)
        i += 1
    if name != "":
        obj = parent.get_children(name)
        if obj is not None:
            p = my_ast.ProcedureCallAST(obj, args)
            p.set_parent(parent)
            return p, i, error
        else:
            error = "Не объявлена процедура с именем " + name
            print(error)
            return None, i, error
    else:
        error = "Требуется имя процедуры"
        print(error)
        return None, i, error


def bin_op_parse(i: int, tokens, parent: my_ast.BaseAST):
    error = ""
    root = None

    j, rpn, error = get_rpn(i, tokens)
    if error != "":
        print(error)
        return None, i, error

    stack = []
    for k in range(len(rpn)):
        if tokens[rpn[k]].type == 'INT_LIT':
            hs = my_ast.IntLiteralAST(tokens[rpn[k]].value)
            stack.append(hs)
            continue
        elif tokens[rpn[k]].type == 'FLOAT_LIT':
            hs = my_ast.FloatLiteralAST(tokens[rpn[k]].value)
            stack.append(hs)
            continue
        elif tokens[rpn[k]].type == 'ID':
            obj = parent.get_children(tokens[rpn[k]].value)
            if obj is None:
                error = "Переменная с именем " + tokens[rpn[k]].value + " не объявлена."
                print(error)
                return None, rpn[k], error
            else:
                if tokens[rpn[k] + 1].type == 'LPAREN':
                    call_obj, i, error = func_call_parse(rpn[k], tokens, parent)
                    if call_obj is None:
                        error = "Функция с именем " + tokens[i].value + " вызвана некорректно."
                        print(error)
                        return None, i, error
                    else:
                        stack.append(call_obj)
                else:
                    var_def_obj = my_ast.VarDefAST(parent)
                    var_def_obj.set_var_dec(obj)
                    stack.append(var_def_obj)
        elif utils.is_operator(tokens[rpn[k]].type):
            bin_op = my_ast.BinaryAST()
            bin_op.set_op(tokens[rpn[k]].type)
            rhs = stack.pop()
            lhs = stack.pop()
            rhs.set_parent(bin_op)
            lhs.set_parent(bin_op)
            bin_op.set_rhs(rhs)
            bin_op.set_lhs(lhs)
            stack.append(bin_op)
    if len(stack) == 1:
        root = stack.pop()
        root.set_parent(parent)
        return root, j, error


def base_parse(tokens):
    base = my_ast.CompoundExpression(None)
    i = 0
    error = ""
    while i < len(tokens):
        base, i, error = top_expression_parse(i, tokens, base)
        if error != "":
            print(error)
            return None, i, error
        i += 1
    return base, i, error


def func_parse(i, tokens, parent=None):
    func = my_ast.FunctionDefAST(parent)
    error = ""
    while tokens[i].type != 'END':
        if tokens[i].type == 'FUNC':
            i += 1
            continue
        elif tokens[i].type == 'ID':
            obj = parent.get_children(tokens[i].value)
            if obj is not None:
                error = "Переменная с именем " + tokens[i].value + " уже объявлена."
                print(error)
                return None, i, error
            parent.add_child(tokens[i].value, func)
            func.set_name(tokens[i].value)
            i += 1
        elif tokens[i].type == 'LPAREN':
            i += 1
            while tokens[i].type != 'RPAREN':
                if tokens[i].type == 'ID':
                    a = parent.get_children(tokens[i].value)
                    if a is not None:
                        error = "Переменная с именем " + tokens[i].value + " уже объявлена во внешней области видимости."
                        print(error)
                        return None, i, error
                    a = my_ast.VarDecAST(func)
                    a.set_name(tokens[i].value)
                    func.add_arg(a)
                    i += 1
                    if utils.is_type(tokens[i].type):
                        a.set_type(tokens[i].type)
                    else:
                        error = "Не указан тип у переменной с именем " + tokens[i].value + "."
                        print(error)
                        return None, i, error
                i += 1
            i += 1
            continue
        elif utils.is_type(tokens[i].type):
            func.set_type(tokens[i].type)
            i += 1
            continue
        elif tokens[i].type == 'COLON':
            if func.type is None:
                error = "Не указан возвращаемый тип у функции с именем " + func.name + "."
                print(error)
                return None, i, error
            i += 1
            while tokens[i].type != 'END':
                _, i, error = compound_expression_parse(i, tokens, func)
                i += 1
            if error != "":
                print(error)
                return None, i, error
    return func, i, error


def proc_parse(i, tokens, parent=None):
    proc = my_ast.ProcedureDefAST(parent)
    error = ""
    while tokens[i].type != 'END':
        if tokens[i].type == 'PROC':
            i += 1
            continue
        elif tokens[i].type == 'ID':
            obj = parent.get_children(tokens[i].value)
            if obj is not None:
                error = "Переменная с именем " + tokens[i].value + " уже объявлена."
                print(error)
                return None, i, error
            parent.add_child(tokens[i].value, proc)
            proc.set_name(tokens[i].value)
            i += 1
        elif tokens[i].type == 'LPAREN':
            i += 1
            while tokens[i].type != 'RPAREN':
                if tokens[i].type == 'ID':
                    a = parent.get_children(tokens[i].value)
                    if a is not None:
                        error = "Переменная с именем " + tokens[i].value + " уже объявлена во внешней области видимости."
                        print(error)
                        return None, i, error
                    a = my_ast.VarDecAST(proc)
                    a.set_name(tokens[i].value)
                    proc.add_arg(a)
                    i += 1
                    if utils.is_type(tokens[i].type):
                        a.set_type(tokens[i].type)
                    else:
                        error = "Не указан тип у переменной с именем " + tokens[i].value + "."
                        print(error)
                        return None, i, error
                i += 1
            i += 1
            continue
        elif tokens[i].type == 'COLON':
            i += 1
            while tokens[i].type != 'END':
                _, i, error = compound_expression_parse(i, tokens, proc)
                i += 1
            if error != "":
                print(error)
                return None, i, error
    return proc, i, error


def compound_expression_parse(i, tokens, compound_expression):
    obj, i, error = parse(i, tokens, parent=compound_expression)
    if error != "":
        print(error)
        return obj, i, error
    compound_expression.set_child(obj)
    return compound_expression, i, error


def top_expression_parse(i, tokens, compound_expression):
    obj, i, error = top_parse(i, tokens, parent=compound_expression)
    if error != "":
        print(error)
        return obj, i, error
    compound_expression.set_child(obj)
    return compound_expression, i, error


def parse(i, tokens, parent=None):
    obj = None
    error = ""
    if tokens[i].type == 'VAR':
        obj, i, error = var_parse(i, tokens, parent)
        if obj is None:
            print(error)
            return None, i, error
    elif tokens[i].type == 'SEMI':
        i += 1
    elif tokens[i].type == 'ID':
        if tokens[i + 1].type == 'RPAREN':
            obj, i, error = func_call_parse(i, tokens, parent)
            if obj is None:
                print(error)
                return None, i, error
        elif tokens[i + 1].type == 'EQ':
            assignment = my_ast.AssignmentAST(parent)
            var_dec_obj = parent.get_children(tokens[i].value)
            var_def_obj = my_ast.VarDefAST(parent)
            var_def_obj.set_var_dec(var_dec_obj)
            assignment.set_lval(var_def_obj)
            obj, i, error = bin_op_parse(i + 2, tokens, parent)
            if obj is None:
                print(error)
                return None, i, error
            assignment.set_rval(obj)
            obj = assignment
        else:
            obj, i, error = proc_call_parse(i, tokens, parent)
            if obj is None:
                print(error)
                return None, i, error
    elif tokens[i].type == 'IF':
        obj, i, error = expr_if_parse(i, tokens, parent)
    elif tokens[i].type == 'WHILE':
        obj, i, error = expr_while_parse(i, tokens, parent)
    elif tokens[i].type == 'DO':
        obj, i, error = expr_do_while_parse(i, tokens, parent)
    elif tokens[i].type == 'RETURN':
        i += 1
        obj, i, error = bin_op_parse(i, tokens, parent)
        if obj is None:
            print(error)
            return None, i, error
        if isinstance(parent, my_ast.FunctionDefAST):
            if obj.get_type() == parent.type:
                parent.set_return_value(obj)
                ret_obj = my_ast.ReturnAst(parent)
                ret_obj.set_value(obj)
                if tokens[i].value != ';':
                    i += 1
                return ret_obj, i, error
            else:
                error = "Ожидается возвращаемый тип " + str(parent.type) + " актуальный тип - " + str(obj.get_type())
                print(error)
                return None, i, error
        else:
            error = "Недопустимая конструкция: return в " + type(parent)
            print(error)
            return None, i, error
    return obj, i, error


def top_parse(i, tokens, parent=None):
    obj = None
    error = ""
    if tokens[i].type == 'VAR':
        obj, i, error = var_parse(i, tokens, parent)
        if obj is None:
            print(error)
            return None, i, error
    elif tokens[i].type == 'SEMI':
        i += 1
    elif tokens[i].type == 'FUNC':
        obj, i, error = func_parse(i, tokens, parent)
        if error != "":
            print(error)
            return None, i, error
    elif tokens[i].type == 'PROC':
        obj, i, error = proc_parse(i, tokens, parent)
        if error != "":
            print(error)
            return None, i, error
    return obj, i, error


def expr_if_parse(i, tokens, parent=None):
    error = ""
    if tokens[i].type == 'IF':
        if_ast = my_ast.ExprIfAST(parent=parent)
        orig_if_ast = if_ast
        i += 1
        if tokens[i].type == 'ID' or tokens[i].type == 'INT_LIT' or tokens[i].type == 'FLOAT_LIT':
            obj, i, error = bin_op_parse(i, tokens, if_ast)
            if_ast.set_expression(obj)
        else:
            error = "Ожидается выражение"
            print(error)
            return None, i, error
        if tokens[i].type == 'COLON':
            i += 1
            then_body = my_ast.CompoundExpression(parent=if_ast)
            while tokens[i].type not in ('ELIF','ELSE','END'):
                then_body, i, error = compound_expression_parse(i, tokens, then_body)
                i += 1
            if error != "":
                print(error)
                return None, i, error
            if_ast.set_then(then_body)
            while tokens[i].type not in ('ELSE', 'END'):
                if tokens[i].type == 'ELIF':
                    i += 1
                    else_body = my_ast.CompoundExpression(if_ast)
                    if_ast.set_else(else_body)
                    if_ast = my_ast.ExprIfAST(else_body)
                    else_body.set_child(if_ast)
                    if tokens[i].type == 'ID' or tokens[i].type == 'INT_LIT' or tokens[i].type == 'FLOAT_LIT':
                        obj, i, error = bin_op_parse(i, tokens, if_ast)
                        if_ast.set_expression(obj)
                    else:
                        error = "Ожидается выражение"
                        print(error)
                        return None, i, error
                    if tokens[i].type == 'COLON':
                        i += 1
                        then_body = my_ast.CompoundExpression(parent=if_ast)
                        while tokens[i].type not in ('ELIF','ELSE','END'):
                            then_body, i, error = compound_expression_parse(i, tokens, then_body)
                            i += 1
                        if error != "":
                            print(error)
                            return None, i, error
                        if_ast.set_then(then_body)
                    else:
                        error = "Ожидается двоеточие"
                        print(error)
                        return None, i, error
            if tokens[i].type == 'ELSE':
                i += 1
                if tokens[i].type == 'COLON':
                    i += 1
                    else_body = my_ast.CompoundExpression(parent=if_ast)
                    while tokens[i].type != 'END':
                        else_body, i, error = compound_expression_parse(i, tokens, else_body)
                        i += 1
                    if error != "":
                        print(error)
                        return None, i, error
                    if_ast.set_else(else_body)
                else:
                    error = "Ожидается двоеточие"
                    print(error)
                    return None, i, error
        return orig_if_ast, i, error


def expr_while_parse(i, tokens, parent=None):
    while_expr = None
    error = ""
    while tokens[i].type != 'END':
        if tokens[i].type == 'WHILE':
            while_expr = my_ast.ExprWhileAST(parent)
            i += 1
            continue
        elif (tokens[i].type == 'ID') or (tokens[i].type == 'INT_LIT') or (tokens[i].type == 'FLOAT_LIT'):
            expr = my_ast.BinaryAST(while_expr)
            expr, i, error = bin_op_parse(i, tokens, expr)
            if error != "":
                print(error)
                return None, i, error
            while_expr.set_expression(expr)
        elif tokens[i].type == 'COLON':
            i += 1
            compound_expression = my_ast.CompoundExpression(parent=while_expr)
            while tokens[i].type != 'END':
                compound_expression, i, error = compound_expression_parse(i, tokens, compound_expression)
                i += 1
            if error != "":
                print(error)
                return None, i, error
            # i += 1
            while_expr.set_body(compound_expression)
            break
    return while_expr, i, error



def expr_do_while_parse(i, tokens, parent=None):
    error = ""
    expr_do = None
    while tokens[i].type != 'SEMI':
        if tokens[i].type == 'DO':
            expr_do = my_ast.ExprDoWhileAST(parent)
            compound_expression = my_ast.CompoundExpression(parent=expr_do)
            i += 1
            continue
        else:
            while tokens[i].type != 'WHILE':
                compound_expression, i, error = compound_expression_parse(i, tokens, compound_expression)
                i += 1
            if error != "":
                print(error)
                return None, i, error
            if tokens[i].type == 'WHILE':
                j = i
                while tokens[j].type not in ['SEMI', 'COLON']:
                    j += 1
                if tokens[j].type == 'COLON':
                    compound_expression, i, error = compound_expression_parse(i, tokens, compound_expression)
                    continue
                elif tokens[j].type == 'SEMI':

                    expr = my_ast.BinaryAST(expr_do)
                    expr, i, error = bin_op_parse(i + 1, tokens, expr)
                    if error != "":
                        print(error)
                        return None, i, error
                    expr_do.set_body(compound_expression)
                    expr_do.set_expression(expr)
                    break
    return expr_do, i, error


def print_result(root, i=0):
    print('  ' * i, type(root).__name__)
    if root is None:
        return
    elif isinstance(root, (my_ast.ExprWhileAST, my_ast.ExprDoWhileAST)):
        print_result(root.expression, i + 1)
        print_result(root.body, i + 1)
    elif isinstance(root, my_ast.ExprIfAST):
        print_result(root.expression, i + 1)
        print_result(root.then_body, i + 1)
        print_result(root.else_body, i + 1)
    elif isinstance(root, my_ast.BinaryAST):
        print('  ' * i, root.operator)
        print_result(root.lhs, i + 1)
        print_result(root.rhs, i + 1)
    elif isinstance(root, my_ast.AssignmentAST):
        print_result(root.lval, i + 1)
        print_result(root.rval, i + 1)
    elif isinstance(root, my_ast.VarDecAST):
        print('  ' * i, root.name, root.type)
    elif isinstance(root, my_ast.VarDefAST):
        print('  ' * i, root.var_dec.name)
    elif isinstance(root, (my_ast.IntLiteralAST, my_ast.FloatLiteralAST, my_ast.CharLiteralAST)):
        print('  ' * i, root.value)
    elif isinstance(root, my_ast.FunctionCallAST):
        print('  ' * i, root.func_callee.name, root.args)
    elif isinstance(root, my_ast.ProcedureCallAST):
        print('  ' * i, root.proc_callee.name, root.args)
    elif isinstance(root, my_ast.CompoundExpression):
        if isinstance(root, (my_ast.FunctionDefAST, my_ast.ProcedureDefAST)):
            print('  ' * i, root.name)
        for op in root.order_operations:
            print_result(op, i + 1)


if __name__ == '__main__':
    lexer = Lexer()
    with open("examples\\correct\\1.txt", 'r', encoding='utf-8') as f:
        code = f.read()
        tokens = lexer.lex(code)
        root, i, errors = base_parse(tokens)

        print_result(root)

        binding.initialize()
        binding.initialize_all_targets()
        binding.initialize_all_asmprinters()

        triple = binding.get_default_triple() # 'mips-PC-Linux-GNU'

        module = ir.Module('module')
        module.triple = triple

        target = binding.Target.from_triple(triple)
        target_machine = target.create_target_machine()

        root.code_gen(module)

        llvm_ir = str(module)
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()

        pass_builder = binding.create_pass_manager_builder()
        mod_pass = binding.create_module_pass_manager()

        # pass_builder.opt_level = 2
        # pass_builder.populate(mod_pass)

        mod_pass.add_constant_merge_pass()
        mod_pass.add_dead_arg_elimination_pass()
        mod_pass.add_function_inlining_pass(225)
        mod_pass.add_global_dce_pass()
        mod_pass.add_global_optimizer_pass()
        mod_pass.add_ipsccp_pass()
        mod_pass.add_dead_code_elimination_pass()
        mod_pass.add_cfg_simplification_pass()
        mod_pass.add_gvn_pass()
        mod_pass.add_instruction_combining_pass()
        mod_pass.add_licm_pass()
        mod_pass.add_sccp_pass()
        mod_pass.add_type_based_alias_analysis_pass()
        mod_pass.add_basic_alias_analysis_pass()

        print(mod_pass.run(mod))

        print(str(mod))

        asm = target_machine.emit_assembly(mod)

        print(asm)

        with open("examples\\correct\\1.s", 'w') as asm_file:
            asm_file.write(asm)

        with open("examples\\correct\\1.elf", 'wb') as obj_file:
            obj_file.write(target_machine.emit_object(mod))
