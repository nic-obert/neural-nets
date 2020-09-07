#pragma once


namespace Datasets {

    // dataset base class
    struct Dataset {
        double** set;
        unsigned int size;

        Dataset(unsigned int _size) 
        : size(_size) {}

        virtual void store(const char* fileName) {}

        virtual void load(const char* fileName) {}

    };


    
};
