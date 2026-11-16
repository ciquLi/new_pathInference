// CPython bridge: preserve Python hash/set/dict iteration and numpy comparisons.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <utility>
#include <vector>

namespace {
struct Error {};
struct O {
    PyObject* p;
    explicit O(PyObject* value) : p(value) { if (!p) throw Error{}; }
    ~O() { Py_XDECREF(p); }
    O(const O&) = delete;
    O(O&& value) noexcept : p(value.p) { value.p = nullptr; }
    O& operator=(O&& value) noexcept {
        if(this!=&value){Py_XDECREF(p);p=value.p;value.p=nullptr;}
        return *this;
    }
    PyObject* release() { auto* value=p; p=nullptr; return value; }
    operator PyObject*() const { return p; }
};
void check(int result) { if (result < 0) throw Error{}; }
O attr(PyObject* object,const char* name) { return O(PyObject_GetAttrString(object,name)); }
O item(PyObject* object,PyObject* key) { return O(PyObject_GetItem(object,key)); }
O number(long value) { return O(PyLong_FromLong(value)); }
long integer(PyObject* value) {
    long result=PyLong_AsLong(value);
    if (result==-1 && PyErr_Occurred()) throw Error{};
    return result;
}
bool contains(PyObject* container,PyObject* key) {
    int result=PySequence_Contains(container,key); check(result); return result;
}
bool compare(PyObject* left,PyObject* right,int op) {
    int result=PyObject_RichCompareBool(left,right,op); check(result); return result;
}
O pair(PyObject* a,PyObject* b) { return O(PyTuple_Pack(2,a,b)); }
PyObject* last(PyObject* tuple) { return PyTuple_GET_ITEM(tuple,PyTuple_GET_SIZE(tuple)-1); }
long length(PyObject* value) {
    auto n=PyObject_Length(value); if(n<0) throw Error{}; return n;
}
O append_path(PyObject* path,PyObject* neighbor) {
    Py_ssize_t n=PyTuple_GET_SIZE(path);
    O result(PyTuple_New(n+1));
    for(Py_ssize_t j=0;j<n;j++){
        auto* node=PyTuple_GET_ITEM(path,j); Py_INCREF(node); PyTuple_SET_ITEM(result.p,j,node);
    }
    Py_INCREF(neighbor); PyTuple_SET_ITEM(result.p,n,neighbor);
    return result;
}
long shortest(PyObject* table,PyObject* source,PyObject* destination,long fallback) {
    O key=pair(source,destination);
    PyObject* value=PyDict_GetItemWithError(table,key);
    if(value) return integer(value);
    if(PyErr_Occurred()) throw Error{};
    return fallback;
}
void push(PyObject* list,long value) { O n=number(value); check(PyList_Append(list,n)); }
void push_object(PyObject* list,PyObject* value) { check(PyList_Append(list,value)); }
long rank(PyObject* orders,PyObject* node,PyObject* neighbor) {
    O values=item(orders,node);
    PyObject* positions=PyObject_GetAttrString(values,"positions");
    if(positions){
        O owned(positions);
        PyObject* value=PyDict_GetItemWithError(positions,neighbor);
        if(value) return integer(value);
        if(PyErr_Occurred()) throw Error{};
        return -1;
    }
    if(!PyErr_ExceptionMatches(PyExc_AttributeError)) throw Error{};
    PyErr_Clear();
    Py_ssize_t value=PySequence_Index(values,neighbor);
    if(value<0){
        if(!PyErr_ExceptionMatches(PyExc_ValueError)) throw Error{};
        PyErr_Clear(); return -1;
    }
    return value;
}
void support(PyObject* list,PyObject* path,PyObject* frequencies,bool suffix,PyObject* separator) {
    long n=PyTuple_GET_SIZE(path);
    for(long count=n;count>0;count--){
        if(count==1){push(list,1);push(list,0);continue;}
        O segment(PyTuple_GetSlice(path,suffix?n-count:0,suffix?n:count));
        O key(PyUnicode_Join(separator,segment));
        PyObject* value=PyDict_GetItemWithError(frequencies,key);
        if(value){push(list,count);push_object(list,value);break;}
        if(PyErr_Occurred()) throw Error{};
    }
}
PyObject* features(PyObject*,PyObject* args) {
    try {
        PyObject *sample,*raw;
        if(!PyArg_ParseTuple(args,"OO",&sample,&raw)) return nullptr;
        O paths(PySequence_Fast(raw,"paths must be a sequence"));
        O table=attr(sample,"shortest"),provider=attr(sample,"provider"),peer=attr(sample,"peer");
        O customer=attr(sample,"customer"),core=attr(sample,"core_cluster");
        O previous=attr(sample,"prev_frequency"),next=attr(sample,"next_frequency");
        O linkfreq=attr(sample,"linkfreq"),orders=attr(sample,"linkorder"),relations=attr(sample,"relationship");
        O separator(PyUnicode_FromString("|")),output(PyList_New(0));
        PyObject* degrees[]={provider,peer,customer};
        for(Py_ssize_t index=0;index<PySequence_Fast_GET_SIZE(paths.p);index++){
            auto* path=PySequence_Fast_GET_ITEM(paths.p,index);
            if(!PyTuple_Check(path) || PyTuple_GET_SIZE(path)<2){
                PyErr_SetString(PyExc_ValueError,"expected AS tuple of length >=2"); throw Error{};
            }
            long n=PyTuple_GET_SIZE(path);
            auto* first=PyTuple_GET_ITEM(path,0); auto* end=last(path);
            auto* penultimate=PyTuple_GET_ITEM(path,n-2);
            O result(PyList_New(0));
            push(result,n); long slen=shortest(table,first,end,n); push(result,slen);
            for(auto* node:{end,penultimate})
                for(auto* degree:degrees){O values=item(degree,node);push(result,length(values));}
            if(contains(core,end)){
                support(result,path,next,true,separator);
                O key=pair(end,penultimate); O value=item(linkfreq,key); push_object(result,value);
            }
            push(result,rank(orders,end,penultimate)); push(result,rank(orders,penultimate,end));
            support(result,path,previous,false,separator);
            long counts[]={0,0,0};
            for(long j=0;j<n-1;j++){
                O key=pair(PyTuple_GET_ITEM(path,j),PyTuple_GET_ITEM(path,j+1));
                O relation=item(relations,key);
                if(PyUnicode_CompareWithASCIIString(relation,"c2p")==0) counts[0]++;
                else if(PyUnicode_CompareWithASCIIString(relation,"p2p")==0) counts[1]++;
                else if(PyUnicode_CompareWithASCIIString(relation,"p2c")==0) counts[2]++;
            }
            for(auto count:counts) push(result,count);
            push(result,n-slen); push(result,n-1-shortest(table,first,penultimate,n-1));
            push_object(output,result);
        }
        return output.release();
    } catch(Error&) { return nullptr; }
}

PyObject* expand(PyObject*,PyObject* args) {
    try{
        PyObject *tree,*active;
        if(!PyArg_ParseTuple(args,"OO",&tree,&active))return nullptr;
        O sample=attr(tree,"ms"),table=attr(sample,"shortest"),core=attr(sample,"core_cluster");
        O provider=attr(sample,"provider"),peer=attr(sample,"peer"),customer=attr(sample,"customer");
        O flags=attr(tree,"path_flag"),paths(PyDict_New());
        O current_paths(PySequence_Fast(active,"active must be a sequence"));
        PyObject* relationships[]={provider,peer,customer};
        const char* names[]={"shell1","shell2","shell3","core1","core2","core3"};
        std::vector<O> keys;
        for(auto name:names)keys.emplace_back(PyUnicode_FromString(name));
        for(long relation=0;relation<3;relation++){
            O relation_value=number(relation);
            for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(current_paths.p);i++){
                auto* current=PySequence_Fast_GET_ITEM(current_paths.p,i);
                if(!PyTuple_Check(current)){PyErr_SetString(PyExc_TypeError,"active path must be tuple");throw Error{};}
                long n=PyTuple_GET_SIZE(current);
                O neighbors=item(relationships[relation],last(current));
                if(n>=9)continue;
                O iterator(PyObject_GetIter(neighbors));
                auto* source=PyTuple_GET_ITEM(current,0);
                O flag=item(flags,current);long state=integer(flag);
                O customers=item(customer,last(current));
                while(auto* next=PyIter_Next(iterator)){
                    O neighbor(next);
                    if(contains(current,neighbor))continue;
                    if(n+1-shortest(table,source,neighbor,n+1)>2)continue;
                    if(state!=0 && !contains(customers,neighbor))continue;
                    O right=append_path(current,neighbor);
                    check(PyDict_SetItem(flags,right,relation_value));
                    long slen=shortest(table,source,neighbor,n+1);
                    int bucket=(contains(core,neighbor)?3:0)+(slen<=3?0:slen<=5?1:2);
                    auto* key=keys[bucket].p;
                    PyObject* values=PyDict_GetItemWithError(paths,key);
                    if(!values){
                        if(PyErr_Occurred())throw Error{};
                        O fresh(PyList_New(0));check(PyDict_SetItem(paths,key,fresh));
                        values=PyDict_GetItem(paths,key);
                    }
                    check(PyList_Append(values,right));
                }
                if(PyErr_Occurred())throw Error{};
            }
        }
        return paths.release();
    }catch(Error&){return nullptr;}
}

PyObject* add_paths(PyObject*,PyObject* args){
    try{
        PyObject *tree,*paths,*candidate;
        if(!PyArg_ParseTuple(args,"OOO",&tree,&paths,&candidate))return nullptr;
        O scores=attr(tree,"spread_score"),as_paths=attr(tree,"AS_path"),threshold=attr(tree,"threshold");
        O get_model=attr(tree,"get_model");
        Py_ssize_t pos=0;PyObject *key,*values;
        while(PyDict_Next(paths,&pos,&key,&values)){
            O result(PyObject_CallFunctionObjArgs(get_model,values,key,nullptr));
            O sequence(PySequence_Fast(result,"model scores must be a sequence"));
            if(PyList_GET_SIZE(values)!=PySequence_Fast_GET_SIZE(sequence.p)){
                PyErr_SetString(PyExc_ValueError,"model score count differs");throw Error{};
            }
            for(Py_ssize_t i=0;i<PyList_GET_SIZE(values);i++){
                auto* path=PyList_GET_ITEM(values,i);
                auto* score=PySequence_Fast_GET_ITEM(sequence.p,i);
                check(PyDict_SetItem(scores,path,score));
                O existing=item(as_paths,last(path));
                if(PyList_GET_SIZE(existing.p)==0){
                    check(PyList_Append(existing,path));check(PySet_Add(candidate,path));
                }else{
                    O previous=item(scores,PyList_GET_ITEM(existing.p,0));
                    if(compare(previous,threshold,Py_LE)&&compare(previous,score,Py_LT)){
                        Py_INCREF(path);check(PyList_SetItem(existing,0,path));check(PySet_Add(candidate,path));
                    }else if(compare(previous,threshold,Py_GT)&&compare(score,threshold,Py_GT)){
                        check(PyList_Append(existing,path));check(PySet_Add(candidate,path));
                    }
                }
            }
        }
        Py_INCREF(candidate);return candidate;
    }catch(Error&){return nullptr;}
}

bool smaller(PyObject* customer,PyObject* peer,PyObject* provider,
             PyObject* parent,PyObject* neighbor,PyObject* worst){
    auto* prev=PyList_GET_ITEM(worst,PyList_GET_SIZE(worst)-2);
    auto* end=PyList_GET_ITEM(worst,PyList_GET_SIZE(worst)-1);
    O customers=item(customer,neighbor),peers=item(peer,end),providers=item(provider,end);
    if(contains(customers,parent)&&(contains(peers,prev)||contains(providers,prev)))return true;
    O first_peers=item(peer,neighbor);
    return contains(first_peers,parent)&&contains(providers,prev);
}

PyObject* shortest_stream(PyObject*,PyObject* args){
    try{
        PyObject *sample,*asn;
        if(!PyArg_ParseTuple(args,"OO",&sample,&asn))return nullptr;
        O provider=attr(sample,"provider"),peer=attr(sample,"peer"),customer=attr(sample,"customer");
        O paths=attr(sample,"AS_path"),worst=attr(sample,"worst"),flags=attr(sample,"path_flag");
        PyObject* relationships[]={provider,peer,customer};
        std::vector<O> active;active.emplace_back(PyTuple_Pack(1,asn));
        while(!active.empty()){
            std::vector<O> next;
            for(long relation=0;relation<3;relation++){
                O flag_value=number(relation);
                for(auto& path:active){
                    auto* parent=last(path);
                    O neighbors=item(relationships[relation],parent);
                    O flag=item(flags,path),customers=item(customer,parent);
                    O iterator(PyObject_GetIter(neighbors));
                    while(auto* value=PyIter_Next(iterator)){
                        O neighbor(value);
                        if(compare(neighbor,asn,Py_EQ))continue;
                        if(integer(flag)!=0&&!contains(customers,neighbor))continue;
                        O right=append_path(path,neighbor);
                        O existing=item(paths,neighbor);
                        if(length(existing)==0||PyTuple_GET_SIZE(right.p)<length(existing)){
                            O copy(PySequence_List(right));check(PyDict_SetItem(paths,neighbor,copy));
                            check(PyDict_SetItem(flags,right,flag_value));
                            Py_INCREF(right.p);next.emplace_back(right.p);
                            O prior=item(worst,neighbor);
                            if(length(prior)==0||smaller(customer,peer,provider,parent,neighbor,prior)){
                                O wcopy(PySequence_List(right));check(PyDict_SetItem(worst,neighbor,wcopy));
                            }
                        }else{
                            O prior=item(worst,neighbor);
                            if(smaller(customer,peer,provider,parent,neighbor,prior)){
                                O copy(PySequence_List(right));check(PyDict_SetItem(worst,neighbor,copy));
                                check(PyDict_SetItem(flags,right,flag_value));
                                Py_INCREF(right.p);next.emplace_back(right.p);
                            }
                        }
                    }
                    if(PyErr_Occurred())throw Error{};
                }
            }
            // Python's sort is stable; identical terminal ASNs retain insertion order.
            std::stable_sort(next.begin(),next.end(),[](const O& a,const O& b){
                O na(PyNumber_Long(last(a.p))),nb(PyNumber_Long(last(b.p)));
                return compare(na,nb,Py_LT);
            });
            active=std::move(next);
            if(PyErr_CheckSignals()<0)throw Error{};
        }
        Py_RETURN_NONE;
    }catch(Error&){return nullptr;}
}
PyMethodDef methods[]={
    {"features",features,METH_VARARGS,"Batch feature extraction in original path order."},
    {"expand",expand,METH_VARARGS,"Expand one routing layer preserving Python set iteration."},
    {"add_paths",add_paths,METH_VARARGS,"Apply original model probabilities and threshold comparisons."},
    {"shortest_stream",shortest_stream,METH_VARARGS,"Original shortest traversal using native loops."},
    {nullptr,nullptr,0,nullptr}
};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_pathradar_native",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__pathradar_native(){return PyModule_Create(&module);}
