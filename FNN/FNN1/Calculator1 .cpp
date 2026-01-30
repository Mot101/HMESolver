
#include "Calculator1.hpp"
#include <sstream>
#include <stack>
#include <cctype>

using namespace std ; 

// ---------------------------
// Utilities
// ---------------------------
static int precedence(char op){
    if(op=='+'||op=='-') return 1 ; 
    if(op=='*'||op=='/') return 2 ; 
    return 0 ; 
}
static bool applyOp(stack<double> &values , char op , string &error){
    if(values.size()<2){
        error ="Not enough operands " ; 
        return false ; 
    }
    double b = values.top(); 
    double a= values.top();
    double res = 0.0; 
    switch (op)
    {
        case '+':
            res = a+b ; 
            break;
        case '-':
            res = a-b ; 
            break;
        case '*':
            res = a*b ; 
            break;
        case '/':
            if (b==0.0){
                error = "division by zero " ; 
                return false ; 
            }
            res = a/b ; 
            break;
        default:
            error = "Unknown operator " ; 
            break;
    }
    values.push(res); 
    return true ; 
}

//-----Calculation step -----//
bool tryEvaluateFromString(const string & expr,double &result , string &error){
    stack<double> values; 
    stack<char> ops ;
    error.clear(); 
    for(size_t i = 0;i<expr.size();i++){
        char c =expr[i]; 
        // to skip the spaces 
        if(isspace((unsigned char)c)){
            i++ ; 
            continue ; 
        }

        if(isdigit((unsigned char)c)|| c=='.'){
            size_t start = i ; 
            while (i<expr.size()&& isdigit((unsigned char)expr[i])|| expr[i]=='.'){
                i++ ; 
            }
            double val ; 
            try{
                val=stod(expr.substr(start,i-start)); 
            }catch(...){
                error = "invalid number "; 
                return false ; 
            }
            values.push(val); 
            continue;
        }
        // closing parenthesis 
        if(c==')'){
            while(!ops.empty()&&ops.top()!='('){
                if(!applyOp(values,ops.top(),error)){
                    return false ; 
                }
                ops.pop();
            }
            if (ops.empty())
            {
                error="mismatched parentheses"; 
                return false ; 
            }
            ops.pop(); // we remove the '('
            i++;
            continue ;    
        }
        if(c=='+'||c=='-'||c=='*'||c=='/'){
            if(c=='-'&& i==0||expr[i-1]=='('||expr[i-1]=='+'||expr[i-1]=='-' || expr[i-1]=='*'||expr[i-1]=='/'){
                values.push(0.0); 
            }
            while(!ops.empty()&&precedence(ops.top())>=precedence(c)) {
                if(!applyOp(values,ops.top(),error)){
                    return false ; 
                }
                ops.pop();
            }
            ops.push(c);
            i++;
            continue;
        }
        error=string("Invalid Character : ") + c ; 
        return false ; 
    }
    while (!ops.empty()){
        if(ops.top()=='('||ops.top()==')'){
            error="mismatched parentheses ."; 
            return false ; 
        }
        if(!applyOp(values,ops.top(),error)){
                    return false ; 
        }
        ops.pop();
    }
    if(values.size()!=1){
        error="Bad Expression "; 
        return false ;  
    }
    result=values.top(); 
    return true ; 
}

