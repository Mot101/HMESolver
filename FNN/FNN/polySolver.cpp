#include "PolySolver.hpp"

#include <cctype>
#include <cmath>
#include <sstream>
#include <algorithm>

using std::string;
using std::vector;
using std::complex;

namespace {

// ---------------- polynomial type ----------------
struct Poly {
    // here c[k] is coefficient of x^k
    vector<double> c;
    Poly():c(1, 0.0) {}
    explicit Poly(double a0):c(1, a0) {}
    static Poly X(){
        Poly p; p.c = {0.0, 1.0};
        return p; 
    }
    int deg() const{
        int d = (int)c.size()-1;
        while (d>0 && std::fabs(c[d])<1e-12) d--;
        return d;
    }
    void trim(){
        int d= deg();
        c.resize(d+1);
    }
};

static Poly add(const Poly& a, const Poly& b){
    Poly r;
    r.c.assign(std::max(a.c.size(), b.c.size()), 0.0);
    for (size_t i=0; i<a.c.size(); i++) r.c[i]+=a.c[i];
    for (size_t i=0; i<b.c.size(); i++) r.c[i]+=b.c[i];
    r.trim();
    return r;
}

static Poly subp(const Poly& a, const Poly& b){
    Poly r;
    r.c.assign(std::max(a.c.size(), b.c.size()), 0.0);
    for (size_t i=0; i<a.c.size(); i++) r.c[i]+=a.c[i];
    for (size_t i=0; i<b.c.size(); i++) r.c[i]-=b.c[i];
    r.trim();
    return r;
}

static Poly mul(const Poly& a, const Poly& b){
    Poly r;
    r.c.assign(a.c.size()+b.c.size()-1, 0.0);
    for (size_t i=0; i<a.c.size(); i++)
        for (size_t j=0; j<b.c.size(); j++)
            r.c[i+j]+=a.c[i]*b.c[j];
    r.trim();
    return r;
}

static Poly neg(const Poly& a){
    Poly r=a;
    for (double& v:r.c) v=-v;
    r.trim();
    return r;
}

// ---------------- tokenizer ----------------
enum class TokType { END, NUM, X, PLUS, MINUS, MUL, LP, RP };
struct Tok{
    TokType t;
    double v; // for NUM
};
struct Lexer{
    string s;
    size_t i=0;

    explicit Lexer(string ss):s(std::move(ss)){}
    static bool isNumChar(char c){
        return std::isdigit((unsigned char)c) || c=='.';
    }

    Tok next(){
        while (i<s.size() && std::isspace((unsigned char)s[i])) i++;
        if (i>=s.size()) return {TokType::END, 0.0};

        char c=s[i];
        if (c=='+'){i++; return {TokType::PLUS, 0.0};}
        if (c=='-'){i++; return {TokType::MINUS, 0.0};}
        if (c=='*'){i++; return {TokType::MUL, 0.0};}
        if (c=='('){i++; return {TokType::LP, 0.0};}
        if (c==')'){i++; return {TokType::RP, 0.0};}
        if (c=='x'|| c=='X'){i++; return {TokType::X, 0.0};}

        if (isNumChar(c)){
            size_t j=i;
            while (j<s.size() && isNumChar(s[j])) j++;
            double val=std::stod(s.substr(i, j-i));
            i=j;
            return {TokType::NUM, val};
        }

        // unknown token
        i++;
        return {TokType::END, 0.0};
    }
};

// ---------------- Recursive descent parser ----------------
struct Parser{
    Lexer lex;
    Tok cur;

    explicit Parser(string s):lex(std::move(s)){
        cur=lex.next();
    }

    void eat(TokType t, const char* msg){
        if (cur.t!=t) throw std::runtime_error(msg);
        cur=lex.next();
    }

    Poly parseExpr(){
        Poly left=parseTerm();
        while (cur.t== TokType::PLUS || cur.t== TokType::MINUS){
            TokType op=cur.t;
            cur=lex.next();
            Poly right=parseTerm();
            left=(op==TokType::PLUS) ? add(left, right):subp(left, right);
        }
        return left;
    }

    Poly parseTerm(){
        Poly left=parseFactor();
        while (cur.t==TokType::MUL){
            cur=lex.next();
            Poly right=parseFactor();
            left=mul(left, right);
        }
        return left;
    }

    Poly parseFactor(){
        if (cur.t==TokType::PLUS){
            cur=lex.next();
            return parseFactor();
        }
        if (cur.t==TokType::MINUS){
            cur=lex.next();
            return neg(parseFactor());
        }

        if (cur.t==TokType::NUM){
            double v=cur.v;
            cur=lex.next();
            return Poly(v);
        }

        if (cur.t==TokType::X){
            cur=lex.next();
            return Poly::X();
        }

        if (cur.t==TokType::LP){
            cur=lex.next();
            Poly inside=parseExpr();
            eat(TokType::RP, "missing ')'");
            return inside;
        }
        throw std::runtime_error("bad polynomial syntax");
    }
};

// ---------------- normalization helpers ----------------
static string stripSpaces(const string& in){
    string out;
    out.reserve(in.size());
    for (char c:in) if (!std::isspace((unsigned char)c)) out.push_back(c);
    return out;
}

static string replaceDotsToMulButKeepDecimals(const string& in) {
    return in;
}

static string insertImplicitMul(const string& in) {
    // here we insert '*', for example:
    // 6x -> 6*x
    // x2 -> x*2
    // ) ( -> )*( 
    // ) x -> )*x
    // 2( -> 2*(
    // x( -> x*(
    auto isOp= [](char c){ return c=='+'||c=='-'||c=='*'||c=='='; };
    auto isAtomL= [&](char c){ return std::isdigit((unsigned char)c) || c=='x' || c=='X' || c==')';};
    auto isAtomR = [&](char c){ return std::isdigit((unsigned char)c) || c=='x' || c=='X' || c=='(';};

    string out;
    out.reserve(in.size()*2);

    for (size_t i=0; i<in.size(); i++){
        char c=in[i];
        out.push_back(c);
        if (i+1<in.size()){
            char n=in[i+1];
            // we take into account not inserting around operators or '='
            if (!isOp(c) && !isOp(n)){
                if (isAtomL(c) && isAtomR(n)){
                    if (!(std::isdigit((unsigned char)c) && n=='.')){
                        if (!(c=='.' && std::isdigit((unsigned char)n))){
                            out.push_back('*');
                        }
                    }
                }
            }
        }
    }
    return out;
}

static void splitEquation(const string& in, string& lhs, string& rhs){
    size_t pos=in.find('=');
    if (pos==string::npos) {lhs=in; rhs= "0"; return;}
    lhs= in.substr(0, pos);
    rhs= in.substr(pos+1);
    if (lhs.empty()) lhs="0";
    if (rhs.empty()) rhs="0";
}

// ---------------- solvers ----------------
static vector<complex<double>> solveLinear(double a, double b){
    vector<complex<double>> r;
    if (std::fabs(a)<1e-12) return r;
    r.push_back(complex<double>(-b/a, 0.0));
    return r;
}
static vector<complex<double>> solveQuadratic(double a, double b, double c){
    if (std::fabs(a)<1e-12) return solveLinear(b, c);
    //we are implementing through discriminant
    complex<double> D=complex<double>(b*b-4*a*c,0.0);
    complex<double> sqrtD=std::sqrt(D);
    complex<double> x1=(-b+sqrtD)/(2.0*a);
    complex<double> x2=(-b-sqrtD)/(2.0*a);
    return {x1, x2};
}

static complex<double> cbrtComplex(const complex<double>& z){
    return std::pow(z,1.0/3.0);
}
static vector<complex<double>> solveCubic(double a, double b, double c, double d){
    if (std::fabs(a)<1e-12) return solveQuadratic(b,c,d);
    double invA=1.0/a;
    double bb=b*invA;
    double cc=c*invA;
    double dd=d*invA;
    double p=cc-(bb*bb)/3.0;
    double q=2.0*(bb*bb*bb)/27.0-(bb*cc)/3.0+dd;

    complex<double> Q= complex<double>(q/2.0,0.0);
    complex<double> P= complex<double>(p/3.0,0.0);
    complex<double> Delta=Q*Q+P*P*P;

    complex<double> sqrtDelta=std::sqrt(Delta);
    complex<double> u=cbrtComplex(-Q+sqrtDelta);
    complex<double> v=cbrtComplex(-Q-sqrtDelta);

    // cube roots of unity
    complex<double> w1(-0.5,std::sqrt(3.0)/2.0);
    complex<double> w2(-0.5,-std::sqrt(3.0)/2.0);
    complex<double> t1=u+v;
    complex<double> t2=u*w1+v*w2;
    complex<double> t3=u*w2+v*w1;
    complex<double> shift=complex<double>(bb/3.0,0.0);
    return {t1-shift,t2-shift,t3-shift};
}

} 

bool trySolvePolynomial(
    const std::string& expr,
    std::vector<std::complex<double>>& roots,
    std::string& normalized,
    std::string& err
){
    roots.clear();
    err.clear();
    normalized.clear();

    try {
        string s = stripSpaces(expr);
        s = replaceDotsToMulButKeepDecimals(s);
        s = insertImplicitMul(s);
        string lhs, rhs;
        splitEquation(s, lhs, rhs);
        // we reject division for now
        if (lhs.find('/')!=string::npos||rhs.find('/')!=string::npos){
            err="polynomial solver does not support division '/'";
            return false;
        }
        normalized=lhs+"=("+rhs+")";
        // we move everything to LHS: lhs-rhs
        Parser pL(lhs);
        Poly PL=pL.parseExpr();
        Parser pR(rhs);
        Poly PR=pR.parseExpr();
        Poly P= subp(PL, PR);
        P.trim();

        int deg=P.deg();
        if (deg==0){
            if (std::fabs(P.c[0])<1e-12){
                err="identity equation (infinite solutions)";
            }else {
                err = "no solution";
            }
            return false;
        }
        // our program cannot handle ploynomials with deg>3 :(
        if (deg>3){
            err="degree>3 not supported (yet)";
            return false;
        }
        // when it solves:
        if (deg==1){
            roots=solveLinear(P.c[1], P.c[0]);
        } else if (deg==2){
            roots=solveQuadratic(P.c[2], P.c[1], P.c[0]);
        } else if (deg==3){
            roots=solveCubic(P.c[3], P.c[2], P.c[1], P.c[0]);
        }
        return !roots.empty();
    } catch (const std::exception& e){
        err=e.what();
        return false;
    }
}
