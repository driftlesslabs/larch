
import ast
import numpy



def inXd(a,*b,**c):
	return numpy.in1d(a,*b,**c).reshape(a.shape)


class AstWrapper(ast.NodeTransformer):

	def visit_Compare(self, node):
		"Change the `in` operator to numpy.in1d"
		if len(node.ops)==1 and isinstance(node.ops[0], ast.In):
			#in1d = ast.Attribute(value=ast.Name(id='numpy', ctx=ast.Load()), attr='in1d', ctx=ast.Load())
			in1d = ast.Name(id='inXd', ctx=ast.Load())
			return ast.Call(in1d, [node.left, node.comparators[0]], [])
		elif len(node.ops)==1 and isinstance(node.ops[0], ast.NotIn):
			#in1d = ast.Attribute(value=ast.Name(id='numpy', ctx=ast.Load()), attr='in1d', ctx=ast.Load())
			in1d = ast.Name(id='inXd', ctx=ast.Load())
			return ast.Call(in1d, [node.left, node.comparators[0]], [ast.keyword('invert',ast.NameConstant(True))])
		else:
			return node


def asterize(cmd, mode="eval"):
	tree = ast.parse(cmd, mode=mode)
	tree = AstWrapper().visit(tree)
	# Add lineno & col_offset to the nodes we created
	ast.fix_missing_locations(tree)
	co = compile(tree, "<ast>", mode)
	return co

