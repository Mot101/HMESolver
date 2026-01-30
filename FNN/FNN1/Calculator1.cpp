#include "Calculator1.hpp"
#include <stack>
#include <cctype>
#include <algorithm>
#include <cmath>
using namespace std;

// for operator priority
static int precedence(char op){
    if(op=='+'||op=='-') return 1;
    if(op=='*'||op=='/') return 2;
    return 0;
}

// we apply one operation to top values on stack
static bool applyOp(stack<double>& values,char op,string& error){
    if(values.size()<2){error="Not enough operands";return false;}
    double b=values.top();values.pop();
    double a=values.top();values.pop();
    double res=0.0;
    switch(op){
        case '+':res=a+b;break;
        case '-':res=a-b;break;
        case '*':res=a*b;break;
        case '/':
            if(b==0.0){
                error="division by zero";
                return false;}
            res=a/b;break;
        default:error="Unknown operator";
        return false;
    }
    values.push(res);
    return true;
}

// evaluate arithmetic expression without parentheses nesting limits
bool tryEvaluateExpression(const string& expr,double& result,string& error){
    stack<double> values;   // numbers
    stack<char> ops;       // operators
    error.clear();

    for(size_t i=0;i<expr.size();i++){
        char c=expr[i];

        // we skip spaces
        if(isspace((unsigned char)c)) continue;

        if(isdigit((unsigned char)c)||c=='.'){
            size_t start=i;
            while(i<expr.size()&&(isdigit((unsigned char)expr[i])||expr[i]=='.')) i++;
            double val;
            try{val=stod(expr.substr(start,i-start));}
            catch(...){error="invalid number";return false;}
            values.push(val);
            i--; // compensate for for-loop increment
            continue;
        }

        // for opening parenthesis (we added it even if we did not include brackets on main)
        if(c=='('){
            ops.push(c);
            continue;
        }

        // for closing parenthesis
        if(c==')'){
            while(!ops.empty()&&ops.top()!='('){
                if(!applyOp(values,ops.top(),error)) return false;
                ops.pop();
            }
            if(ops.empty()){error="mismatched parentheses";return false;}
            ops.pop();
            continue;
        }

        // operator handling 
        if(c=='+'||c=='-'||c=='*'||c=='/'){
            if(c=='-'&&(i==0||expr[i-1]=='('||expr[i-1]=='+'||expr[i-1]=='-'||expr[i-1]=='*'||expr[i-1]=='/'))
                values.push(0.0);
            while(!ops.empty()&&precedence(ops.top())>=precedence(c)){
                if(!applyOp(values,ops.top(),error)) return false;
                ops.pop();
            }
            ops.push(c);
            continue;
        }
        error=string("Invalid Character: ")+c;
        return false;
    }
    while(!ops.empty()){
        if(ops.top()=='('||ops.top()==')'){error="mismatched parentheses";return false;}
        if(!applyOp(values,ops.top(),error)) return false;
        ops.pop();
    }
    
    // final result check
    if(values.size()!=1){error="Bad Expression";return false;}
    result=values.top();
    return true;
}
