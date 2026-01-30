#include "Calculator.hpp"
#include <stack>
#include <cctype>
#include <sstream>

using namespace std;

// ---------------- helpers ----------------

static int precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/') return 2;
    return 0;
}

static bool applyOp(stack<double>& values, char op, string& error) {
    if (values.size() < 2) {
        error = "not enough operands";
        return false;
    }

    double b = values.top(); values.pop();
    double a = values.top(); values.pop();

    double res = 0.0;
    switch (op) {
        case '+': res=a+b; break;
        case '-': res=a-b; break;
        case '*': res=a*b; break;
        case '/':
            if (b == 0.0) {
                error = "division by zero";
                return false;
            }
            res=a/b;
            break;
        default:
            error = "unknown operator";
            return false;
    }

    values.push(res);
    return true;
}

// ---------------- main evaluator ----------------

bool tryEvaluateExpression(
    const string& expr,
    double& result,
    string& error
) {
    stack<double> values;
    stack<char> ops;

    error.clear();

    for (size_t i = 0; i < expr.size(); ) {
        char c = expr[i];

        // Skip spaces
        if (isspace((unsigned char)c)) {
            i++;
            continue;
        }

        // Number (supports decimals)
        if (isdigit((unsigned char)c) || c == '.') {
            size_t start = i;
            while (i < expr.size() &&
                  (isdigit((unsigned char)expr[i]) || expr[i] == '.')) {
                i++;
            }
            double val;
            try {
                val = stod(expr.substr(start, i - start));
            } catch (...) {
                error = "invalid number";
                return false;
            }
            values.push(val);
            continue;
        }

        // Opening parenthesis
        if (c == '(') {
            ops.push(c);
            i++;
            continue;
        }

        // Closing parenthesis
        if (c == ')') {
            while (!ops.empty() && ops.top() != '(') {
                if (!applyOp(values, ops.top(), error)) return false;
                ops.pop();
            }
            if (ops.empty()) {
                error = "mismatched parentheses";
                return false;
            }
            ops.pop(); // remove '('
            i++;
            continue;
        }

        // Operator
        if (c == '+' || c == '-' || c == '*' || c == '/') {
            // Unary minus handling: (-5) or start with -5
            if (c == '-' &&
                (i == 0 || expr[i-1] == '(' ||
                 expr[i-1] == '+' || expr[i-1] == '-' ||
                 expr[i-1] == '*' || expr[i-1] == '/')) {
                values.push(0.0);
            }

            while (!ops.empty() &&
                   precedence(ops.top()) >= precedence(c)) {
                if (!applyOp(values, ops.top(), error)) return false;
                ops.pop();
            }

            ops.push(c);
            i++;
            continue;
        }

        error = string("invalid character: ") + c;
        return false;
    }

    // Final reductions
    while (!ops.empty()) {
        if (ops.top() == '(' || ops.top() == ')') {
            error = "mismatched parentheses";
            return false;
        }
        if (!applyOp(values, ops.top(), error)) return false;
        ops.pop();
    }

    if (values.size() != 1) {
        error = "bad expression";
        return false;
    }

    result = values.top();
    return true;
}
