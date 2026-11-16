// Stream the large frequency corpus directly into immutable C++ evidence.
#define PyInit__pathradar_pipeline PyInit__pathradar_pipeline_embedded
#include "pipeline_native.cpp"
#undef PyInit__pathradar_pipeline
#include <fstream>
#include <stdexcept>

namespace {
PyObject* pack_file(PyObject* self,PyObject* args){
    try{
        PyObject *sample,*filename;
        if(!PyArg_ParseTuple(args,"OO",&sample,&filename))return nullptr;
        // All mutations occur before publishing the capsule to the caller.
        O capsule(pack(self,sample));
        auto* evidence=static_cast<Evidence*>(PyCapsule_GetPointer(capsule,EVIDENCE));if(!evidence)throw Error{};
        std::ifstream stream(string(filename));
        if(!stream){PyErr_SetString(PyExc_OSError,"cannot open PathRadar join frequency file");throw Error{};}
        std::string line;size_t row=0;
        while(std::getline(stream,line)){
            ++row;
            // Match Python strip().split(' ') for the ASCII routing corpus.
            auto start=line.find_first_not_of(" \t\r\n\v\f");auto end=line.find_last_not_of(" \t\r\n\v\f");
            if(start==std::string::npos){PyErr_SetString(PyExc_ValueError,"blank frequency record");throw Error{};}
            auto a=line.find(' ',start),b=a==std::string::npos?a:line.find(' ',a+1);
            if(a==std::string::npos||b==std::string::npos||b>=end||line.find(' ',b+1)<=end){
                PyErr_SetString(PyExc_ValueError,"expected three frequency fields");throw Error{};
            }
            auto flag=line.substr(start,a-start),key=line.substr(a+1,b-a-1),raw=line.substr(b+1,end-b);
            size_t used=0;int64_t value=0;
            try{value=std::stoll(raw,&used);}catch(const std::exception&){PyErr_SetString(PyExc_ValueError,"invalid int64 frequency");throw Error{};}
            if(used!=raw.size()){PyErr_SetString(PyExc_ValueError,"invalid frequency integer");throw Error{};}
            (flag=="prev"?evidence->previous:evidence->next)[std::move(key)]=value;
            if(row%65536==0&&PyErr_CheckSignals()<0)throw Error{};
        }
        if(stream.bad()){PyErr_SetString(PyExc_OSError,"frequency corpus read failed");throw Error{};}
        return capsule.release();
    }catch(Error&){return nullptr;}catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
PyObject* evidence_equal(PyObject*,PyObject* args){
    try{
        PyObject *left,*right;
        if(!PyArg_ParseTuple(args,"OO",&left,&right))return nullptr;
        auto* a=static_cast<Evidence*>(PyCapsule_GetPointer(left,EVIDENCE));if(!a)throw Error{};
        auto* b=static_cast<Evidence*>(PyCapsule_GetPointer(right,EVIDENCE));if(!b)throw Error{};
        if(a->previous!=b->previous||a->next!=b->next||a->nodes.size()!=b->nodes.size()||a->edges.size()!=b->edges.size())Py_RETURN_FALSE;
        std::vector<Id> translate;
        for(const auto& node:a->nodes){
            Id id=b->find(node.text);
            if(id==MISSING||node.degree!=b->nodes[id].degree||node.core!=b->nodes[id].core)Py_RETURN_FALSE;
            translate.push_back(id);
        }
        for(const auto& item:a->edges){
            auto old=item.first;
            auto found=b->edges.find(edge(translate[old>>32],translate[old&0xffffffff]));
            if(found==b->edges.end())Py_RETURN_FALSE;
            const auto& x=item.second;const auto& y=found->second;
            if(x.relation!=y.relation||x.rank!=y.rank||x.frequency!=y.frequency)Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    }catch(Error&){return nullptr;}catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
PyMethodDef loader_methods[]={
    {"pack_file",pack_file,METH_VARARGS,"Load frequencies without intermediate Python dictionaries."},
    {"evidence_equal",evidence_equal,METH_VARARGS,"Compare every evidence value across independent loaders."},
    {"distances",distances,METH_VARARGS,"Pack unchanged distances."},
    {"features",features,METH_VARARGS,"Construct unchanged feature matrices."},
    {"pack_adjacency",pack_adjacency,METH_VARARGS,"Snapshot adjacency in original order."},
    {"expand",expand_ordered,METH_VARARGS,"Expand through integer tables."},
    {"add_paths",update_paths,METH_VARARGS,"Preserve original NumPy comparisons."},
    {nullptr,nullptr,0,nullptr}};
PyModuleDef loader_module={PyModuleDef_HEAD_INIT,"_pathradar_loader",nullptr,-1,loader_methods};
}
PyMODINIT_FUNC PyInit__pathradar_loader(){return PyModule_Create(&loader_module);}
