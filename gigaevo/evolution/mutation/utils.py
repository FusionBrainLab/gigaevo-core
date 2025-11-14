import ast


class _DocstringRemover(ast.NodeTransformer):
    """AST transformer that removes docstrings from Python code."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Remove docstrings from function definitions."""
        self._remove_docstring(node)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        """Remove docstrings from async function definitions."""
        self._remove_docstring(node)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Remove docstrings from class definitions."""
        self._remove_docstring(node)
        self.generic_visit(node)
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Remove module-level docstrings."""
        self._remove_docstring(node)
        self.generic_visit(node)
        return node

    def _remove_docstring(self, node):
        """Remove docstring from a node if it exists."""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            # Check if it's actually a string constant (docstring)
            if isinstance(node.body[0].value, ast.Constant) and isinstance(
                node.body[0].value.value, str
            ):
                node.body.pop(0)
            elif isinstance(node.body[0].value, ast.Str):
                node.body.pop(0)
