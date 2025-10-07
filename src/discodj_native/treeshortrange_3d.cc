#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdint.h>

#include <iterator>
#include <algorithm>
#include <random>
#include <thread>
#include <iostream>

// #define STANDALONE

namespace py = pybind11;

#if defined(__BMI2__)
#include <immintrin.h>
#endif

//! 64bit mask for 3D Morton indexing
static constexpr uint64_t morton_code_3d{0b001001001001001001001001001001001001001001001001001001001001001001}; // 3*21 bits
//! use 21 bits per dimension for Morton encoding
static constexpr uint64_t morton_range_3d{1ull<<20};


//! encode a 3D cartesian coordinate in Morton
/*!
    \param x array of three Cartesian coordinates in [0,boxsize)x[0,boxsize)x[0,boxsize)
    \return 64 bit Morton index
 */
uint64_t xyz_to_morton (const std::array<double,3>& x, const double boxsize)
{
  #if defined(__BMI2__)
    return _pdep_u64( static_cast<uint64_t>(x[0] / boxsize * morton_range_3d), morton_code_3d<<0 )
         | _pdep_u64( static_cast<uint64_t>(x[1] / boxsize * morton_range_3d), morton_code_3d<<1 )
         | _pdep_u64( static_cast<uint64_t>(x[2] / boxsize * morton_range_3d), morton_code_3d<<2 );
  #else
    #warning native CPU does not support BMI2, using fallback implementation
    uint64_t res = 0;
    for( int dim=0; dim<3; ++dim ){
      auto mask = morton_code_3d << dim;
      const auto xx   = static_cast<uint64_t>(x[dim] / boxsize * morton_range_3d);
      for (uint64_t bb = 1; mask != 0; bb += bb) {
        if (xx & bb) { res |= mask & (-mask); }
        mask &= (mask - 1);
      }
    }
    return res;
  #endif
}

//! decode a 3D cartesian coordinate from Morton code
/*!
    \param m 64 bit Morton index
    \param x reference to array of three Cartesian coordinates in [0,boxsize)x[0,boxsize)x[0,boxsize) where result is returned
 */
void morton_to_xyz(uint64_t m, std::array<double,3>& x, double boxsize)
{
  #if defined(__BMI2__)
    x[0] = double(_pext_u64(m, morton_code_3d<<0))/morton_range_3d*boxsize;
    x[1] = double(_pext_u64(m, morton_code_3d<<1))/morton_range_3d*boxsize;
    x[2] = double(_pext_u64(m, morton_code_3d<<2))/morton_range_3d*boxsize;
  #else
    for( int dim=0; dim<3; ++dim )
    {
      uint64_t res = 0;
      auto mask = morton_code_3d<<dim;
      for (uint64_t bb = 1; mask != 0; bb += bb) {
        if (m & mask & -mask) { res |= bb; }
        mask &= (mask - 1);
      }
      x[dim] = double(res)/morton_range_3d*boxsize;
    }
  #endif
}

//! Kronecker delta, return 1 if i=j otherwise 0
inline double Kronecker(int i, int j){
    return (i==j)? 1.0 : 0.0;
}

//! correct 1D distance on circle
inline double periodic_d(double d, double boxsize){
    double boxsize_half = boxsize / 2.0;
    return (d<-boxsize_half)? d+boxsize : ((d>boxsize_half)? d-boxsize : d);
}

//! make sure coordinate is periodically mapped into [0, boxsize)
inline double periodic_wrap(double x, double boxsize){
    return std::fmod(x + boxsize, boxsize);
}

//! class for Morton ordered 3D vectors
class ordered_vec3_t{

protected:
    std::array<double,3> v_;    //!< cartesian vector components
    uint64_t ID_;               //!< ID attached to vector (e.g. array index)
    uint64_t order_;            //!< Morton index

public:
    //! vector component access operator [write]
    double& operator[](int i){ return v_[i]; }
    //! vector component access operator [read]
    const double& operator[](int i) const { return v_[i]; }

    //! comparison operator [const]
    bool operator<( const ordered_vec3_t& o ) const{
        return order_ < o.order_;
    }

    //! comparison operator [non-const]
    bool operator<( const uint64_t& mkey ) const{
        return order_ < mkey;
    }

    //! return Morton key
    uint64_t get_key() const{
        return order_;
    }

    //! return associated ID
    uint64_t get_ID() const{
        return ID_;
    }

    //! constructor
    ordered_vec3_t(std::array<double,3>&& v, uint64_t ID, double boxsize)
    : v_( std::move(v) ), ID_( ID ), order_( xyz_to_morton(v_, boxsize) ) {}
};

//! class storing multipole expansions
struct multipole_t{
    using particle_vt = std::vector< ordered_vec3_t >; //!< type of the particle container
    double M;                               //!< monopole moment
    std::array<double,3> D;                 //!< dipole moment
    std::array<std::array<double,3>,3> Q;   //!< quadrupole moment

    //! zero all multipole moments
    void zero()
    {
        M = 0.0;
        D.fill( 0.0 );
        for(int i=0;i<3;++i)
            Q[i].fill(0.0);
    }

    //! constructor -- does nothing
    multipole_t()
    {
        this->zero();
    }

    //! compute the multipole moments for a set of particles
    /*!
        \param particles const reference to the particle container
        \param il lower bound of the particle index range
        \param ir upper bound of the particle index range
        \param center geometric center around which the expansion is made
     */
    void add_P2M(const particle_vt& particles, const uint64_t il, const uint64_t ir, const std::array<double,3>& center,
     const double boxsize)
    {
        for( uint64_t i=il; i<ir; ++i ){
            const std::array<double,3> rc{{
                periodic_d(particles[i][0] - center[0], boxsize),
                periodic_d(particles[i][1] - center[1], boxsize),
                periodic_d(particles[i][2] - center[2], boxsize) }};
            const double rcnorm2 = rc[0]*rc[0] + rc[1]*rc[1] + rc[2]*rc[2];

            M += 1.0;
            for( int i1=0; i1<3; ++i1 ){
                double r1 = rc[i1];
                D[i1] += r1;
                for( int i2=0; i2<3; ++i2 ){
                    double r2 = rc[i2];
                    Q[i1][i2] += 3.0 * r1 * r2 - Kronecker(i1,i2) * rcnorm2;
                }
            }
        }
    }
};

//! Tree node type class
struct node_t
{
    using particle_vt = std::vector< ordered_vec3_t >; //!< particle container type
    using node_p = node_t*;                            //!< node pointer type
    uint64_t il_, ir_;                                 //!< index of left and right particle of each node
    std::array<node_p,8> childp_;                      //!< array containing pointers to the 8 children of a node
    char level_;                                       //!< tree refinement of level of a node
    std::array<double,3> center_;                      //!< geometric center of the node
    multipole_t M_;                                    //!< multipole expansion of the node mass
    double boxsize_;                                   //!< boxsize

    //! compute interaction between a particle and the node
    /*!
        \param particles const reference to the particle container
        \param ip index of the particle for which to compute the interactions
        \param theta opening angle used for the force acceleration
        \param alpha2 square of the short-range/long range cutoff scale
        \param acc reference to a C++ array to which the resulting acceleration is written
     */
    void accumulate_acc(const particle_vt& particles, uint64_t ip, double theta, double alpha2, std::array<double,3>& acc){
        const std::array<double,3> dv{{
            periodic_d(center_[0]-particles[ip][0], boxsize_),
            periodic_d(center_[1]-particles[ip][1], boxsize_),
            periodic_d(center_[2]-particles[ip][2], boxsize_)}};

        constexpr double eps{1e-10}, eps2{eps*eps};
        const double d2 = std::max(dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2],eps2);
        const double d  = std::sqrt(d2);
        const double ell = boxsize_ / (1ull<<level_);
        const double alpha = std::sqrt(alpha2);

        if( ell < theta * d)
        {
            const double d4 = d2*d2;
            const double d6 = d4*d2;
            const double d3 = d2 * std::sqrt(d2);
            const double d5 = d2 * d3;

#if defined(STANDALONE)
            for( int i=0; i<3; ++i ){
                // monopole
                acc[i]  += dv[i] / d3 * this->M_.M; // TODO: mass scaling in standalone version!!!
                for( int j=0; j<3; ++j ){
                    // dipole
                    acc[i] -= (3 * dv[i] * dv[j] - Kronecker(i,j) * d2)/d5 * this->M_.D[j];
                    // for( int k=0; k<3; ++k ){
                    //     // quadrupole
                    //     acc[i] += 0.5 * ( 4 * dv[i]*dv[j]*dv[k] - (Kronecker(i,j)*dv[k] + Kronecker(i,k)*dv[j])*d2 )/d6 * M_.Q[j][k];
                    // }
                }
            }
#else
            const std::array<double,3> dx{{ periodic_d(center_[0]+this->M_.D[0]/this->M_.M-particles[ip][0], boxsize_),
                                            periodic_d(center_[1]+this->M_.D[1]/this->M_.M-particles[ip][1], boxsize_),
                                            periodic_d(center_[2]+this->M_.D[2]/this->M_.M-particles[ip][2], boxsize_) }};
            const double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
            const double r  = std::sqrt(r2);
            const double r3 = r2 * r;

            const double W  = std::erfc(r / (std::sqrt(2.0) * alpha)) + std::sqrt(2.0/M_PI) * r / alpha * std::exp(-0.5 * r2 / alpha2);


            for( int i=0; i<3; ++i ){
                // monopole
                // acc[i]  += dv[i] / d2 * this->M_.M * W;

                // shifted monopole

                acc[i]  += dx[i] / r3 * this->M_.M * W * boxsize_ * boxsize_ * boxsize_;

                // for( int j=0; j<2; ++j ){
                //     // dipole
                //     acc[i] -= (2 * dv[i] * dv[j] - Kronecker(i,j) * d2)/d4 * this->M_.D[j];
                //     for( int k=0; k<2; ++k ){
                //         // quadrupole
                //         acc[i] += 0.5 * ( 4 * dv[i]*dv[j]*dv[k] - (Kronecker(i,j)*dv[k] + Kronecker(i,k)*dv[j])*d2 )/d6 * M_.Q[j][k];
                //     }
                // }
            }
#endif


        }else{
            bool hasnochildren = true;
            for( auto cp : childp_ ){
                if( cp != nullptr ){ // then we can recursively call the tree walk
                    cp->accumulate_acc( particles, ip, theta, alpha2, acc );
                    hasnochildren = false;
                }
            }
            if( hasnochildren ) // then do P2P summation of node-associated particles
            {
                for( size_t jp=il_; jp<ir_; ++jp ){
                    if( ip==jp ) continue;
                    const double dx = periodic_d(particles[jp][0] - particles[ip][0], boxsize_);
                    const double dy = periodic_d(particles[jp][1] - particles[ip][1], boxsize_);
                    const double dz = periodic_d(particles[jp][2] - particles[ip][2], boxsize_);
#if defined(STANDALONE)
                    const double r2 = dx*dx+dy*dy+dz*dz;
                    const double r3 = r2 * std::sqrt(r2);
                    acc[0] += dx / r3 * boxsize_ * boxsize_ * boxsize_;
                    acc[1] += dy / r3 * boxsize_ * boxsize_ * boxsize_;
                    acc[2] += dz / r3 * boxsize_ * boxsize_ * boxsize_;
#else

                    const double r2 = dx*dx+dy*dy+dz*dz;
                    const double r  = std::sqrt(r2);
                    const double r3 = r2 * r;
                    const double W  = std::erfc(r / (std::sqrt(2.0) * alpha)) + std::sqrt(2.0/M_PI) * r / alpha * std::exp(-0.5 * r2 / alpha2);

                    acc[0] += dx / r3 * W * boxsize_ * boxsize_ * boxsize_;
                    acc[1] += dy / r3 * W * boxsize_ * boxsize_ * boxsize_;
                    acc[2] += dz / r3 * W * boxsize_ * boxsize_ * boxsize_;
#endif
                }
            }
        }
    }

    //! constructor for a node
    /*!
        \param particles const reference to the particle container
        \param il index of first particle in node, i.e. il<=node particles<ir
        \param ir index beyond last particle in node, i.e. il<=node particles<ir
        \param ml lower bound morton index of the node
        \param mr upper bound morton index of the node
        \param level refinement level on which node sits
        \param maxnodesize_ maximum number of particles in node before it will get refined
        \param boxsize size of the simulation box
     */
    node_t( const particle_vt& particles, const uint64_t il, uint64_t ir, uint64_t ml, uint64_t mr, char level, const uint64_t maxnodesize_, double boxsize)
    : il_(il), ir_(ir), childp_{nullptr}, level_(level), boxsize_(boxsize)
    {
        morton_to_xyz( ml, center_, boxsize_);
        center_[0] += 0.5 * boxsize_/(1ull<<level);
        center_[1] += 0.5 * boxsize_/(1ull<<level);
        center_[2] += 0.5 * boxsize_/(1ull<<level);

        M_.add_P2M( particles, il, ir, center_, boxsize_ );

        if( ir_-il_ > maxnodesize_ )
        {
            auto dm = (mr - ml)/8;
            for( uint64_t subd=0; subd<8; ++subd ){
                auto cml = ml + subd * dm;
                auto cmr = ml + (subd+1) * dm;
                auto cil = std::distance<const ordered_vec3_t*>( &particles[0], std::lower_bound( &particles[il], &particles[ir], cml ) );
                auto cir = std::distance<const ordered_vec3_t*>( &particles[0], std::lower_bound( &particles[il], &particles[ir], cmr ) );
                if( cir-cil > 0 )
                {
                    childp_[subd] = new node_t( particles, cil, cir, cml, cmr, level_+1, maxnodesize_, boxsize_);
                }
            }
            //M_.add_P2M( particles, il, ir, center_ ); // normally should optimize to do M2M upwards here, since the child multipoles already exist!
        }else{
            //M_.add_P2M( particles, il, ir, center_ );
        }
    }

    //! copy constructor [deleted]
    node_t( const node_t& ) = delete;
    //! move constructor [deleted]
    node_t( const node_t&& ) = delete;

    //! destructor
    ~node_t()
    {
        for( uint64_t subd=0; subd<8; ++subd ){
            if( childp_[subd] != nullptr ) delete childp_[subd];
        }
    }
};

//! DD N-body solver class
struct nbody3d
{
    using particle_vt = std::vector< ordered_vec3_t >; //!< type of the particle container
    particle_vt particles_; //!< particle container
    node_t* rootnode_;      //!< pointer to the tree root node
    double boxsize_;        //!< size of the simulation box
    //! constructor
    /*! initialise the N-body solver from a Python numpy array of particle positions
        \param input the input particle positions
        \return none
     */
    explicit nbody3d( const py::array_t<double>& input, double boxsize )
    : rootnode_(nullptr), boxsize_(boxsize)
    {
        py::buffer_info buf = input.request();

        auto bForder = input.attr("flags")["F_CONTIGUOUS"].cast<bool>();
        auto bCorder = input.attr("flags")["C_CONTIGUOUS"].cast<bool>();

        if (buf.ndim != 2 || buf.shape[1]!=3 || (!bForder&&!bCorder))
            throw std::runtime_error("Number of dimensions must be two for 3D sort and input array must be non-sparse!");

        /* No pointer is passed, so NumPy will allocate the buffer */
        auto result = py::array_t<double>(buf.size);

        py::buffer_info buf_r = result.request();

        double *ptr = static_cast<double *>(buf.ptr);
        if( bForder ){
            for (auto idx = 0; idx < buf.shape[0]; idx++){
                particles_.emplace_back( ordered_vec3_t(
                    { periodic_wrap(ptr[idx], boxsize_),
                      periodic_wrap(ptr[idx+buf.shape[0]], boxsize_),
                      periodic_wrap(ptr[idx+2*buf.shape[0]], boxsize_) },
                    idx, boxsize_ ) );
            }
        }else{
            for (auto idx = 0; idx < buf.shape[0]; idx++){
                particles_.emplace_back( ordered_vec3_t(
                    { periodic_wrap(ptr[3*idx+0], boxsize_),
                      periodic_wrap(ptr[3*idx+1], boxsize_),
                      periodic_wrap(ptr[3*idx+2], boxsize_) },
                    idx, boxsize_ ) );
            }
        }

        std::sort( particles_.begin(), particles_.end() );

        //
        rootnode_ = new node_t( particles_, 0, particles_.size(), 0, 1ull<<60, 0, 32, boxsize_ );
        // rootnode_ = new node_t( particles_, 0, particles_.size(), 0, uint64_t(-1), 0, 32, boxsize_ );
    }

    //! get the particle positions as a numpy array
    /*!
        \return python numpy array of size (N,3) with the particle positions
     */
    py::array_t<double> get_pos( void ) const
    {
        auto result = py::array_t<double>(3 * particles_.size() );

        py::buffer_info buf_r = result.request();

        double *ptr = static_cast<double *>(buf_r.ptr);

        for (size_t idx = 0; idx < particles_.size(); idx++)
        {
            ptr[3*idx+0] = particles_[idx][0];
            ptr[3*idx+1] = particles_[idx][1];
            ptr[3*idx+2] = particles_[idx][2];
        }

        result.resize({int(particles_.size()),3});

        return result;
    }

    //! Split a number most evenly
    /*!
        \param numitems the total number of items to distribute
        \param numchunks the number of chunks to divide numitems into
        \param it output iterator where the number items in each chunk will be written
        \return none
     */
    template< typename OutputIterator >
    void chunkify( size_t numitems, size_t numchunks, OutputIterator it ) const
    {
        size_t minitems = numitems / numchunks;
        size_t remainder = numitems % numchunks;

        size_t count = 0;
        for( size_t ichunk = 0; ichunk < numchunks; ++ichunk ){
            size_t items = minitems + (( ichunk < remainder )? 1 : 0);
            *it++ = count;
            count += items;
        }
        *it = count;
    }

    //! Compute particle-particle interactions
    /*! Used as a 'compute kernel' in multi-threaded evaluations
        \param alpha the short-range/long-range cutoff scale
        \param ip the index of the particle for which to compute the interactions
        \param acc reference to a C++ std::array where the result will be stored
        \return no return value
     */
    void compute_acc_P2P( const double alpha, const uint64_t ip, std::array<double,3>& acc ) const
    {
        const double alpha2 = alpha*alpha;
        acc.fill( 0.0 );
        for( size_t jp=0; jp<particles_.size(); ++jp ){
            if( ip==jp ) continue;
#if defined(STANDALONE)
            double dx = particles_[jp][0] - particles_[ip][0];
            double dy = particles_[jp][1] - particles_[ip][1];
            double dz = particles_[jp][2] - particles_[ip][2];
            double r2 = dx*dx+dy*dy+dz*dz;
            double r3 = r2 * std::sqrt(r2);
            acc[0] += dx / r3 * boxsize_ * boxsize_;
            acc[1] += dy / r3 * boxsize_ * boxsize_;
            acc[2] += dz / r3 * boxsize_ * boxsize_;
#else
            double dx = periodic_d(particles_[jp][0] - particles_[ip][0], boxsize_);
            double dy = periodic_d(particles_[jp][1] - particles_[ip][1], boxsize_);
            double dz = periodic_d(particles_[jp][2] - particles_[ip][2], boxsize_);
            double r2 = dx*dx+dy*dy+dz*dz;
            double r  = std::sqrt(r2);
            double r3 = r2 * r;

            const double W  = std::erfc(r / (std::sqrt(2.0) * alpha)) + std::sqrt(2.0/M_PI) * r / alpha * std::exp(-0.5 * r2 / alpha2);
            acc[0] += dx / r3 * W  * boxsize_ * boxsize_ * boxsize_;
            acc[1] += dy / r3 * W  * boxsize_ * boxsize_ * boxsize_;
            acc[2] += dz / r3 * W  * boxsize_ * boxsize_ * boxsize_;
#endif
        }
    }

    //! Perform a tree walk to get multipole-particle interactions
    /*! Used as a 'compute kernel' in multi-threaded evaluations
        \param theta the tree 'opening angle', controls the multipole approximation error
        \param alpha the short-range/long-range cutoff scale
        \param ip the index of the particle for which to compute the interactions
        \param acc reference to a C++ std::array where the result will be stored
        \return no return value
     */
    void compute_acc_tree( const double theta, const double alpha, const uint64_t ip, std::array<double,3>& acc) const
    {
        acc.fill( 0.0 );
        this->rootnode_->accumulate_acc( particles_, ip, theta, alpha*alpha, acc );
    }

    //! Compute the interaction between a particle and a particle
    /*! Compute particle-particle interaction using multiple threads. The particles are
        split into chunks according to the number of threads available, then subsequently
        direct summation is performed
        \param alpha the short-range/long-range cutoff scale
        \return returns a python numpy array with the accelerations for each particle
     */
    py::array_t<double> get_acc_P2P( const double alpha=0.1 ) const
    {
        const auto nump = particles_.size();
        const auto numthreads = std::thread::hardware_concurrency();
        // const auto numthreads = 1; // for debugging
        auto result = py::array_t<double>(3 * nump );

        py::buffer_info buf_r = result.request();

        double *ptr = static_cast<double *>(buf_r.ptr);

        std::vector<std::thread> threadlist;
        std::vector<uint64_t> threadchunks;

        this->chunkify( nump, numthreads, std::back_inserter( threadchunks ) );

        for( size_t ithread=0; ithread<numthreads; ++ithread ){
            threadlist.emplace_back( std::thread( [this, ithread, &threadchunks, ptr, alpha, nump](){
                std::array<double,3> acc;
                const double fourpi = 4.0*M_PI;
                for( size_t ip=threadchunks[ithread]; ip<threadchunks[ithread+1]; ++ip )
                {
                    // direct summation over all particles
                    this->compute_acc_P2P( alpha, ip, acc );
                    auto ipp = particles_[ip].get_ID();
                    ptr[3*ipp+0] = -acc[0] / fourpi / nump;
                    ptr[3*ipp+1] = -acc[1] / fourpi / nump;
                    ptr[3*ipp+2] = -acc[2] / fourpi / nump;
                }
            }) );
        }

        for( auto &t : threadlist ) t.join();

        result.resize({int(particles_.size()),3});

        return result;
    }

    //! Compute the interaction between a multipole and a particle
    /*! Compute multipole-particle interaction using multiple threads. The particles are
        split into chunks according to the number of threads available, then subsequently
        a tree walk is executed for each particle in the chunk.
        \param theta the tree 'opening angle', controls the multipole approximation error
        \param alpha the short-range/long-range cutoff scale
        \return returns a python numpy array with the accelerations for each particle
     */
    py::array_t<double> get_acc_M2P(const double theta=0.7, const double alpha=0.1) const
    {
        const auto nump = particles_.size();
        const auto numthreads = std::thread::hardware_concurrency();

        auto result = py::array_t<double>(3 * nump);

        py::buffer_info buf_r = result.request();

        double *ptr = static_cast<double *>(buf_r.ptr);

        std::vector<std::thread> threadlist;
        std::vector<uint64_t> threadchunks;

        this->chunkify(nump, numthreads, std::back_inserter(threadchunks));

        for(size_t ithread=0; ithread<numthreads; ++ithread){
            threadlist.emplace_back( std::thread( [this, ithread, &threadchunks, nump, ptr, theta, alpha](){
                std::array<double,3> acc;
                const double fourpi = 4.0*M_PI;
                for( size_t ip=threadchunks[ithread]; ip<threadchunks[ithread+1]; ++ip )
                {
                    // direct summation over all particles
                    this->compute_acc_tree( theta, alpha, ip, acc);
                    auto ipp = particles_[ip].get_ID();
                    ptr[3*ipp+0] = -acc[0] / fourpi / nump;
                    ptr[3*ipp+1] = -acc[1] / fourpi / nump;
                    ptr[3*ipp+2] = -acc[2] / fourpi / nump;
                }
            }) );
        }

        for( auto &t : threadlist ) t.join();

        result.resize({int(particles_.size()),3});

        return result;
    }

    //! output number of particles in the container
    void print_stat() const{
        py::print("nbody3d contains ",particles_.size()," particles.");
        py::print("Boxsize: ", boxsize_);
    }
};

void init_nbody3d(py::module &m)
{
    py::class_<nbody3d>(m, "nbody3d")
        .def(py::init<const py::array_t<double> &, double>())
        .def("get_pos", &nbody3d::get_pos)
        .def("get_acc_M2P", &nbody3d::get_acc_M2P)
        .def("get_acc_P2P", &nbody3d::get_acc_P2P)
        .def("print_stat", &nbody3d::print_stat);
}
