/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Modified by Carnegie Mellon University, 2026: seam anchors, corridor-cell
    metadata, setup options and debug snapshots added to GCOPTER::GCOPTER_PolytopeSFC.
*/

#ifndef GCOPTER_HPP
#define GCOPTER_HPP

#include "minco.hpp"
#include "flatness.hpp"
#include "lbfgs.hpp"
#include "geo_utils.hpp"

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <vector>

namespace gcopter
{

    enum class CellType : uint8_t
    {
        MOTION = 0,
        VISIBILITY = 1,
    };

    struct SeamAnchor
    {
        bool enabled = false;
        int splitPolyIndex = -1;
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        double yaw = 0.0;
        double positionWeight = 0.0;
        double yawWeight = 0.0;
    };

    struct CorridorCellMeta
    {
        CellType type = CellType::MOTION;
        bool locked = false;
        int minPieces = 1;
    };

    struct SetupOptions
    {
        std::vector<CorridorCellMeta> cellMeta;
        int observationCellIndex = -1;
        double observationYaw = 0.0;
        double observationYawWeight = 0.0;
        SeamAnchor seamAnchor;
    };

    struct DebugSnapshot
    {
        int polyNum = 0;
        int pieceNum = 0;
        double allocSpeed = 0.0;

        bool hasSeamAnchor = false;
        int seamPieceIndex = -1;
        Eigen::Vector3d seamPosition = Eigen::Vector3d::Zero();
        double seamYaw = 0.0;

        Eigen::Matrix3Xd shortPath;
        Eigen::VectorXi pieceIdx;
        Eigen::VectorXi vPolyIdx;
        Eigen::VectorXi hPolyIdx;
        std::vector<int> cellTypes;
        int observationCellIndex = -1;
        int observationKnotIndex = -1;

        Eigen::Matrix3Xd initialPoints;
        Eigen::VectorXd initialTimes;
        Eigen::VectorXd initialYawKnots;

        Eigen::Matrix3Xd finalPoints;
        Eigen::VectorXd finalTimes;
        Eigen::VectorXd finalYawKnots;

        double costEnergy = 0.0;
        double costTime = 0.0;
        double costPos = 0.0;
        double costVel = 0.0;
        double costOmg = 0.0;
        double costTheta = 0.0;
        double costThrust = 0.0;
        double costYawSmooth = 0.0;
        double costYawRateLimit = 0.0;
        double costSeamPos = 0.0;
        double costSeamYaw = 0.0;
        double costObservationYaw = 0.0;
    };

    class GCOPTER_PolytopeSFC
    {
    public:
        typedef Eigen::Matrix3Xd PolyhedronV;
        typedef Eigen::MatrixX4d PolyhedronH;
        typedef std::vector<PolyhedronV> PolyhedraV;
        typedef std::vector<PolyhedronH> PolyhedraH;

    private:
        minco::MINCO_S3NU minco;
        flatness::FlatnessMap flatmap;

        double rho;
        double jerkWeight = 1.0;
        Eigen::Matrix3d headPVA;
        Eigen::Matrix3d tailPVA;

        PolyhedraV vPolytopes;
        PolyhedraH hPolytopes;
        Eigen::Matrix3Xd shortPath;

        Eigen::VectorXi pieceIdx;
        Eigen::VectorXi vPolyIdx;
        Eigen::VectorXi hPolyIdx;

        int polyN;
        int pieceN;

        int spatialDim;
        int temporalDim;

        double smoothEps;
        int integralRes;
        Eigen::VectorXd magnitudeBd;
        Eigen::VectorXd penaltyWt;
        Eigen::VectorXd physicalPm;
        double allocSpeed;
        double initMaxAcc;
        double initTimeMargin;

        lbfgs::lbfgs_parameter_t lbfgs_params;

        Eigen::Matrix3Xd points;
        Eigen::VectorXd times;
        Eigen::Matrix3Xd gradByPoints;
        Eigen::VectorXd gradByTimes;
        Eigen::VectorXd yawKnots;
        Eigen::VectorXd gradByYawKnots;
        Eigen::MatrixX3d partialGradByCoeffs;
        Eigen::VectorXd partialGradByTimes;
        double startYaw;
        double goalYaw;
        double yawSmoothWeight;
        double maxYawRate;
        double yawRatePenaltyWeight;
        Eigen::VectorXd initialYawGuessKnots;
        std::vector<CorridorCellMeta> cellMeta;
        bool hasSeamAnchor = false;
        int seamPieceIndex = -1;
        Eigen::Vector3d seamAnchorPosition = Eigen::Vector3d::Zero();
        double seamAnchorYaw = 0.0;
        double seamAnchorPositionWeight = 0.0;
        double seamAnchorYawWeight = 0.0;
        bool hasObservationCell = false;
        int observationCellIndex = -1;
        int observationKnotIndex = -1;
        double observationYaw = 0.0;
        double observationYawWeight = 0.0;
        Eigen::Matrix3Xd initialPointsSnapshot;
        Eigen::VectorXd initialTimesSnapshot;
        Eigen::VectorXd initialYawKnotsSnapshot;
        double lastCostEnergy = 0.0;
        double lastCostTime = 0.0;
        double lastCostPos = 0.0;
        double lastCostVel = 0.0;
        double lastCostOmg = 0.0;
        double lastCostTheta = 0.0;
        double lastCostThrust = 0.0;
        double lastCostYawSmooth = 0.0;
        double lastCostYawRateLimit = 0.0;
        double lastCostSeamPos = 0.0;
        double lastCostSeamYaw = 0.0;
        double lastCostObservationYaw = 0.0;

    private:
        static inline void forwardT(const Eigen::VectorXd &tau,
                                    Eigen::VectorXd &T)
        {
            const int sizeTau = tau.size();
            T.resize(sizeTau);
            for (int i = 0; i < sizeTau; i++)
            {
                T(i) = tau(i) > 0.0
                           ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                           : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
            }
            return;
        }

        template <typename EIGENVEC>
        static inline void backwardT(const Eigen::VectorXd &T,
                                     EIGENVEC &tau)
        {
            const int sizeT = T.size();
            tau.resize(sizeT);
            for (int i = 0; i < sizeT; i++)
            {
                tau(i) = T(i) > 1.0
                             ? (sqrt(2.0 * T(i) - 1.0) - 1.0)
                             : (1.0 - sqrt(2.0 / T(i) - 1.0));
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void backwardGradT(const Eigen::VectorXd &tau,
                                         const Eigen::VectorXd &gradT,
                                         EIGENVEC &gradTau)
        {
            const int sizeTau = tau.size();
            gradTau.resize(sizeTau);
            double denSqrt;
            for (int i = 0; i < sizeTau; i++)
            {
                if (tau(i) > 0)
                {
                    gradTau(i) = gradT(i) * (tau(i) + 1.0);
                }
                else
                {
                    denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
                    gradTau(i) = gradT(i) * (1.0 - tau(i)) / (denSqrt * denSqrt);
                }
            }

            return;
        }

        static inline double unwrapAngleNear(const double reference,
                                             const double angle)
        {
            return reference + std::remainder(angle - reference, 2.0 * M_PI);
        }

        template <typename EIGENVEC>
        static inline void forwardYaw(const EIGENVEC &eta,
                                      const double &psiStart,
                                      const double &psiGoal,
                                      Eigen::VectorXd &psi)
        {
            psi.resize(eta.size() + 2);
            psi(0) = psiStart;
            if (eta.size() > 0)
            {
                psi.segment(1, eta.size()) = eta;
            }
            psi(psi.size() - 1) = psiGoal;
            return;
        }

        template <typename EIGENVEC>
        static inline void backwardYaw(const Eigen::VectorXd &psi,
                                       EIGENVEC &eta)
        {
            const int n = std::max<int>(psi.size() - 2, 0);
            eta.resize(n);
            if (n > 0)
            {
                eta = psi.segment(1, n);
            }
            return;
        }

        static inline void setInitialYaw(const double &psiStart,
                                         const double &psiGoal,
                                         const Eigen::VectorXd &initialYawGuess,
                                         Eigen::VectorXd &psi)
        {
            if (initialYawGuess.size() >= 2)
            {
                psi = initialYawGuess;
                psi(0) = psiStart;
                psi(psi.size() - 1) = psiGoal;
                return;
            }

            psi.resize(2);
            psi(0) = psiStart;
            psi(1) = psiGoal;
            return;
        }

        static inline void buildInitialYawGuess(const Eigen::Matrix3Xd &path,
                                                const Eigen::VectorXi &pieceIdx,
                                                const double &psiStart,
                                                const double &psiGoal,
                                                Eigen::VectorXd &psiGuess)
        {
            const int polyNum = pieceIdx.size();
            const int pieceNum = pieceIdx.sum();
            psiGuess.resize(pieceNum + 1);
            psiGuess.setConstant(psiStart);
            if (polyNum <= 0 || path.cols() < polyNum + 1)
            {
                if (psiGuess.size() >= 1)
                {
                    psiGuess(psiGuess.size() - 1) = psiGoal;
                }
                return;
            }

            Eigen::VectorXd segLengths(polyNum);
            double totalLength = 0.0;
            for (int i = 0; i < polyNum; ++i)
            {
                segLengths(i) = (path.col(i + 1) - path.col(i)).norm();
                totalLength += segLengths(i);
            }

            if (totalLength <= 1.0e-9)
            {
                for (int i = 0; i <= pieceNum; ++i)
                {
                    const double frac = pieceNum > 0 ? static_cast<double>(i) / static_cast<double>(pieceNum) : 1.0;
                    psiGuess(i) = psiStart + frac * (psiGoal - psiStart);
                }
                return;
            }

            psiGuess(0) = psiStart;
            int knotIdx = 0;
            double accumLength = 0.0;
            for (int i = 0; i < polyNum; ++i)
            {
                const int subPieces = std::max(pieceIdx(i), 1);
                const double segLength = segLengths(i);
                for (int j = 0; j < subPieces; ++j)
                {
                    const double fracWithinSeg = static_cast<double>(j + 1) / static_cast<double>(subPieces);
                    const double s = accumLength + fracWithinSeg * segLength;
                    const double frac = std::clamp(s / totalLength, 0.0, 1.0);
                    psiGuess(++knotIdx) = psiStart + frac * (psiGoal - psiStart);
                }
                accumLength += segLength;
            }
            psiGuess(pieceNum) = psiGoal;
            return;
        }

        static inline void buildInitialYawGuessWithSeam(const Eigen::VectorXi &pieceIdx,
                                                        const double &psiStart,
                                                        const double &psiSeam,
                                                        const double &psiGoal,
                                                        const int seamPieceIdx,
                                                        Eigen::VectorXd &psiGuess)
        {
            const int pieceNum = pieceIdx.sum();
            psiGuess.resize(pieceNum + 1);
            if (pieceNum <= 0)
            {
                psiGuess.resize(1);
                psiGuess(0) = psiStart;
                return;
            }

            const double seamYaw = unwrapAngleNear(psiStart, psiSeam);
            const double goalYawFromSeam = unwrapAngleNear(seamYaw, psiGoal);
            const int seamIdx = std::clamp(seamPieceIdx, 1, pieceNum - 1);

            psiGuess(0) = psiStart;
            for (int i = 1; i <= seamIdx; ++i)
            {
                const double frac = static_cast<double>(i) / static_cast<double>(seamIdx);
                psiGuess(i) = psiStart + frac * (seamYaw - psiStart);
            }
            for (int i = seamIdx + 1; i <= pieceNum; ++i)
            {
                const double frac =
                    static_cast<double>(i - seamIdx) / static_cast<double>(pieceNum - seamIdx);
                psiGuess(i) = seamYaw + frac * (goalYawFromSeam - seamYaw);
            }
            psiGuess(pieceNum) = goalYawFromSeam;
            return;
        }

        static inline void buildInitialYawGuessWithObservation(const Eigen::VectorXi &pieceIdx,
                                                               const double &psiStart,
                                                               const double &psiObservation,
                                                               const double &psiGoal,
                                                               const int observationCellIdx,
                                                               Eigen::VectorXd &psiGuess)
        {
            const int pieceNum = pieceIdx.sum();
            psiGuess.resize(pieceNum + 1);
            if (pieceNum <= 0)
            {
                psiGuess.resize(1);
                psiGuess(0) = psiStart;
                return;
            }

            const int polyNum = pieceIdx.size();
            if (observationCellIdx < 0 || observationCellIdx >= polyNum)
            {
                buildInitialYawGuessWithSeam(pieceIdx,
                                             psiStart,
                                             psiObservation,
                                             psiGoal,
                                             std::clamp(pieceNum / 2, 1, std::max(pieceNum - 1, 1)),
                                             psiGuess);
                return;
            }

            const double obsYaw = unwrapAngleNear(psiStart, psiObservation);
            const double goalYawFromObs = unwrapAngleNear(obsYaw, psiGoal);
            const int obsStartKnot = pieceIdx.head(observationCellIdx).sum();
            const int obsEndKnot = pieceIdx.head(observationCellIdx + 1).sum();

            auto ease_out = [](const double u)
            {
                const double uc = std::clamp(u, 0.0, 1.0);
                const double one_minus = 1.0 - uc;
                return 1.0 - one_minus * one_minus;
            };

            psiGuess(0) = psiStart;

            if (obsStartKnot <= 0)
            {
                psiGuess(0) = obsYaw;
            }
            else
            {
                for (int i = 0; i <= obsStartKnot; ++i)
                {
                    const double frac = static_cast<double>(i) / static_cast<double>(obsStartKnot);
                    const double shaped = ease_out(frac);
                    psiGuess(i) = psiStart + shaped * (obsYaw - psiStart);
                }
            }

            for (int i = std::max(obsStartKnot, 0); i <= std::min(obsEndKnot, pieceNum); ++i)
            {
                psiGuess(i) = obsYaw;
            }

            if (obsEndKnot >= pieceNum)
            {
                psiGuess(pieceNum) = goalYawFromObs;
                return;
            }

            const int tailKnots = pieceNum - obsEndKnot;
            for (int i = obsEndKnot; i <= pieceNum; ++i)
            {
                const double frac =
                    tailKnots > 0 ? static_cast<double>(i - obsEndKnot) / static_cast<double>(tailKnots) : 1.0;
                psiGuess(i) = obsYaw + frac * (goalYawFromObs - obsYaw);
            }
            psiGuess(pieceNum) = goalYawFromObs;
            return;
        }

        static inline void forwardP(const Eigen::VectorXd &xi,
                                    const Eigen::VectorXi &vIdx,
                                    const PolyhedraV &vPolys,
                                    Eigen::Matrix3Xd &P)
        {
            const int sizeP = vIdx.size();
            P.resize(3, sizeP);
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();
                q = xi.segment(j, k).normalized().head(k - 1);
                P.col(i) = vPolys[l].rightCols(k - 1) * q.cwiseProduct(q) +
                           vPolys[l].col(0);
            }
            return;
        }

        static inline double costTinyNLS(void *ptr,
                                         const Eigen::VectorXd &xi,
                                         Eigen::VectorXd &gradXi)
        {
            const int n = xi.size();
            const Eigen::Matrix3Xd &ovPoly = *(Eigen::Matrix3Xd *)ptr;

            const double sqrNormXi = xi.squaredNorm();
            const double invNormXi = 1.0 / sqrt(sqrNormXi);
            const Eigen::VectorXd unitXi = xi * invNormXi;
            const Eigen::VectorXd r = unitXi.head(n - 1);
            const Eigen::Vector3d delta = ovPoly.rightCols(n - 1) * r.cwiseProduct(r) +
                                          ovPoly.col(1) - ovPoly.col(0);

            double cost = delta.squaredNorm();
            gradXi.head(n - 1) = (ovPoly.rightCols(n - 1).transpose() * (2 * delta)).array() *
                                 r.array() * 2.0;
            gradXi(n - 1) = 0.0;
            gradXi = (gradXi - unitXi.dot(gradXi) * unitXi).eval() * invNormXi;

            const double sqrNormViolation = sqrNormXi - 1.0;
            if (sqrNormViolation > 0.0)
            {
                double c = sqrNormViolation * sqrNormViolation;
                const double dc = 3.0 * c;
                c *= sqrNormViolation;
                cost += c;
                gradXi += dc * 2.0 * xi;
            }

            return cost;
        }

        template <typename EIGENVEC>
        static inline void backwardP(const Eigen::Matrix3Xd &P,
                                     const Eigen::VectorXi &vIdx,
                                     const PolyhedraV &vPolys,
                                     EIGENVEC &xi)
        {
            const int sizeP = P.cols();

            double minSqrD;
            lbfgs::lbfgs_parameter_t tiny_nls_params;
            tiny_nls_params.past = 0;
            tiny_nls_params.delta = 1.0e-5;
            tiny_nls_params.g_epsilon = FLT_EPSILON;
            tiny_nls_params.max_iterations = 128;

            Eigen::Matrix3Xd ovPoly;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();

                ovPoly.resize(3, k + 1);
                ovPoly.col(0) = P.col(i);
                ovPoly.rightCols(k) = vPolys[l];
                Eigen::VectorXd x(k);
                x.setConstant(sqrt(1.0 / k));
                lbfgs::lbfgs_optimize(x,
                                      minSqrD,
                                      &GCOPTER_PolytopeSFC::costTinyNLS,
                                      nullptr,
                                      nullptr,
                                      &ovPoly,
                                      tiny_nls_params);

                xi.segment(j, k) = x;
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void backwardGradP(const Eigen::VectorXd &xi,
                                         const Eigen::VectorXi &vIdx,
                                         const PolyhedraV &vPolys,
                                         const Eigen::Matrix3Xd &gradP,
                                         EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();
            gradXi.resize(xi.size());

            double normInv;
            Eigen::VectorXd q, gradQ, unitQ;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();
                q = xi.segment(j, k);
                normInv = 1.0 / q.norm();
                unitQ = q * normInv;
                gradQ.resize(k);
                gradQ.head(k - 1) = (vPolys[l].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradXi.segment(j, k) = (gradQ - unitQ * unitQ.dot(gradQ)) * normInv;
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void normRetrictionLayer(const Eigen::VectorXd &xi,
                                               const Eigen::VectorXi &vIdx,
                                               const PolyhedraV &vPolys,
                                               double &cost,
                                               EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();
            gradXi.resize(xi.size());

            double sqrNormQ, sqrNormViolation, c, dc;
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k; i < sizeP; i++, j += k)
            {
                k = vPolys[vIdx(i)].cols();

                q = xi.segment(j, k);
                sqrNormQ = q.squaredNorm();
                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradXi.segment(j, k) += dc * 2.0 * q;
                }
            }

            return;
        }

        static inline bool smoothedL1(const double &x,
                                      const double &mu,
                                      double &f,
                                      double &df)
        {
            if (x < 0.0)
            {
                return false;
            }
            else if (x > mu)
            {
                f = x - 0.5 * mu;
                df = 1.0;
                return true;
            }
            else
            {
                const double xdmu = x / mu;
                const double sqrxdmu = xdmu * xdmu;
                const double mumxd2 = mu - 0.5 * x;
                f = mumxd2 * sqrxdmu * xdmu;
                df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
                return true;
            }
        }

        // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
        // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
        // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
        //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
        static inline void attachPenaltyFunctional(const Eigen::VectorXd &T,
                                                   const Eigen::MatrixX3d &coeffs,
                                                   const Eigen::VectorXi &hIdx,
                                                   const PolyhedraH &hPolys,
                                                   const double &smoothFactor,
                                                   const int &integralResolution,
                                                   const Eigen::VectorXd &magnitudeBounds,
                                                   const Eigen::VectorXd &penaltyWeights,
                                                   flatness::FlatnessMap &flatMap,
                                                   double &cost,
                                                   double &costPos,
                                                   double &costVel,
                                                   double &costOmg,
                                                   double &costTheta,
                                                   double &costThrust,
                                                   Eigen::VectorXd &gradT,
                                                   Eigen::MatrixX3d &gradC)
        {
            const double velSqrMax = magnitudeBounds(0) * magnitudeBounds(0);
            const double omgSqrMax = magnitudeBounds(1) * magnitudeBounds(1);
            const double thetaMax = magnitudeBounds(2);
            const double thrustMean = 0.5 * (magnitudeBounds(3) + magnitudeBounds(4));
            const double thrustRadi = 0.5 * fabs(magnitudeBounds(4) - magnitudeBounds(3));
            const double thrustSqrRadi = thrustRadi * thrustRadi;

            const double weightPos = penaltyWeights(0);
            const double weightVel = penaltyWeights(1);
            const double weightOmg = penaltyWeights(2);
            const double weightTheta = penaltyWeights(3);
            const double weightThrust = penaltyWeights(4);

            Eigen::Vector3d pos, vel, acc, jer, sna;
            Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
            double totalGradPsi, totalGradPsiD;
            double thr, cos_theta;
            Eigen::Vector4d quat;
            Eigen::Vector3d omg;
            double gradThr;
            Eigen::Vector4d gradQuat;
            Eigen::Vector3d gradPos, gradVel, gradOmg;

            double step, alpha;
            double s1, s2, s3, s4, s5;
            Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;
            Eigen::Vector3d outerNormal;
            int K, L;
            double violaPos, violaVel, violaOmg, violaTheta, violaThrust;
            double violaPosPenaD, violaVelPenaD, violaOmgPenaD, violaThetaPenaD, violaThrustPenaD;
            double violaPosPena, violaVelPena, violaOmgPena, violaThetaPena, violaThrustPena;
            double sampleCostPos, sampleCostVel, sampleCostOmg, sampleCostTheta, sampleCostThrust;
            double node, pena;

            const int pieceNum = T.size();
            const double integralFrac = 1.0 / integralResolution;
            for (int i = 0; i < pieceNum; i++)
            {
                const Eigen::Matrix<double, 6, 3> &c = coeffs.block<6, 3>(i * 6, 0);
                step = T(i) * integralFrac;
                for (int j = 0; j <= integralResolution; j++)
                {
                    s1 = j * step;
                    s2 = s1 * s1;
                    s3 = s2 * s1;
                    s4 = s2 * s2;
                    s5 = s4 * s1;
                    beta0(0) = 1.0, beta0(1) = s1, beta0(2) = s2, beta0(3) = s3, beta0(4) = s4, beta0(5) = s5;
                    beta1(0) = 0.0, beta1(1) = 1.0, beta1(2) = 2.0 * s1, beta1(3) = 3.0 * s2, beta1(4) = 4.0 * s3, beta1(5) = 5.0 * s4;
                    beta2(0) = 0.0, beta2(1) = 0.0, beta2(2) = 2.0, beta2(3) = 6.0 * s1, beta2(4) = 12.0 * s2, beta2(5) = 20.0 * s3;
                    beta3(0) = 0.0, beta3(1) = 0.0, beta3(2) = 0.0, beta3(3) = 6.0, beta3(4) = 24.0 * s1, beta3(5) = 60.0 * s2;
                    beta4(0) = 0.0, beta4(1) = 0.0, beta4(2) = 0.0, beta4(3) = 0.0, beta4(4) = 24.0, beta4(5) = 120.0 * s1;
                    pos = c.transpose() * beta0;
                    vel = c.transpose() * beta1;
                    acc = c.transpose() * beta2;
                    jer = c.transpose() * beta3;
                    sna = c.transpose() * beta4;

                    flatMap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

                    violaVel = vel.squaredNorm() - velSqrMax;
                    violaOmg = omg.squaredNorm() - omgSqrMax;
                    cos_theta = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));
                    cos_theta = std::clamp(cos_theta, -1.0 + 1.0e-9, 1.0 - 1.0e-9);
                    violaTheta = acos(cos_theta) - thetaMax;
                    violaThrust = (thr - thrustMean) * (thr - thrustMean) - thrustSqrRadi;

                    gradThr = 0.0;
                    gradQuat.setZero();
                    gradPos.setZero(), gradVel.setZero(), gradOmg.setZero();
                    pena = 0.0;
                    sampleCostPos = 0.0;
                    sampleCostVel = 0.0;
                    sampleCostOmg = 0.0;
                    sampleCostTheta = 0.0;
                    sampleCostThrust = 0.0;

                    L = hIdx(i);
                    K = hPolys[L].rows();
                    for (int k = 0; k < K; k++)
                    {
                        outerNormal = hPolys[L].block<1, 3>(k, 0);
                        violaPos = outerNormal.dot(pos) + hPolys[L](k, 3);
                        if (smoothedL1(violaPos, smoothFactor, violaPosPena, violaPosPenaD))
                        {
                            gradPos += weightPos * violaPosPenaD * outerNormal;
                            pena += weightPos * violaPosPena;
                            sampleCostPos += weightPos * violaPosPena;
                        }
                    }

                    if (smoothedL1(violaVel, smoothFactor, violaVelPena, violaVelPenaD))
                    {
                        gradVel += weightVel * violaVelPenaD * 2.0 * vel;
                        pena += weightVel * violaVelPena;
                        sampleCostVel += weightVel * violaVelPena;
                    }

                    if (smoothedL1(violaOmg, smoothFactor, violaOmgPena, violaOmgPenaD))
                    {
                        gradOmg += weightOmg * violaOmgPenaD * 2.0 * omg;
                        pena += weightOmg * violaOmgPena;
                        sampleCostOmg += weightOmg * violaOmgPena;
                    }

                    if (smoothedL1(violaTheta, smoothFactor, violaThetaPena, violaThetaPenaD))
                    {
                        const double sin_theta = std::sqrt(std::max(1.0 - cos_theta * cos_theta, 1.0e-9));
                        gradQuat += weightTheta * violaThetaPenaD /
                                    sin_theta * 4.0 *
                                    Eigen::Vector4d(0.0, quat(1), quat(2), 0.0);
                        pena += weightTheta * violaThetaPena;
                        sampleCostTheta += weightTheta * violaThetaPena;
                    }

                    if (smoothedL1(violaThrust, smoothFactor, violaThrustPena, violaThrustPenaD))
                    {
                        gradThr += weightThrust * violaThrustPenaD * 2.0 * (thr - thrustMean);
                        pena += weightThrust * violaThrustPena;
                        sampleCostThrust += weightThrust * violaThrustPena;
                    }

                    flatMap.backward(gradPos, gradVel, gradThr, gradQuat, gradOmg,
                                     totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                                     totalGradPsi, totalGradPsiD);

                    node = (j == 0 || j == integralResolution) ? 0.5 : 1.0;
                    alpha = j * integralFrac;
                    costPos += node * step * sampleCostPos;
                    costVel += node * step * sampleCostVel;
                    costOmg += node * step * sampleCostOmg;
                    costTheta += node * step * sampleCostTheta;
                    costThrust += node * step * sampleCostThrust;
                    gradC.block<6, 3>(i * 6, 0) += (beta0 * totalGradPos.transpose() +
                                                    beta1 * totalGradVel.transpose() +
                                                    beta2 * totalGradAcc.transpose() +
                                                    beta3 * totalGradJer.transpose()) *
                                                   node * step;
                    gradT(i) += (totalGradPos.dot(vel) +
                                 totalGradVel.dot(acc) +
                                 totalGradAcc.dot(jer) +
                                 totalGradJer.dot(sna)) *
                                    alpha * node * step +
                                node * integralFrac * pena;
                    cost += node * step * pena;
                }
            }

            return;
        }

        static inline void attachYawFunctional(const Eigen::VectorXd &T,
                                               const Eigen::VectorXd &psi,
                                               const double &smoothWeight,
                                               const double &yawRateMax,
                                               const double &yawRatePenalty,
                                               const double &smoothFactor,
                                               double &cost,
                                               double &costYawSmooth,
                                               double &costYawRateLimit,
                                               Eigen::VectorXd &gradT,
                                               Eigen::VectorXd &gradPsi)
        {
            const int pieceNum = T.size();
            gradPsi.setZero(psi.size());

            const double yawRateMaxSqr = yawRateMax > 0.0 ? yawRateMax * yawRateMax : INFINITY;
            for (int i = 0; i < pieceNum; ++i)
            {
                const double dt = std::max(T(i), 1.0e-9);
                const double dpsi = psi(i + 1) - psi(i);
                const double invDt = 1.0 / dt;
                const double invDtSqr = invDt * invDt;

                double gradDelta = 0.0;
                if (smoothWeight > 0.0)
                {
                    // Time-normalized yaw smoothness. For piecewise-linear yaw,
                    // this is the integral of squared yaw rate over the segment.
                    const double smoothCost = smoothWeight * dpsi * dpsi * invDt;
                    cost += smoothCost;
                    costYawSmooth += smoothCost;
                    gradDelta += 2.0 * smoothWeight * dpsi * invDt;
                    gradT(i) += -smoothWeight * dpsi * dpsi * invDtSqr;
                }

                if (yawRatePenalty > 0.0 && std::isfinite(yawRateMaxSqr))
                {
                    const double rateSqr = dpsi * dpsi * invDtSqr;
                    const double violation = rateSqr - yawRateMaxSqr;
                    double pena = 0.0;
                    double penaD = 0.0;
                    if (smoothedL1(violation, smoothFactor, pena, penaD))
                    {
                        const double yawRateCost = yawRatePenalty * dt * pena;
                        cost += yawRateCost;
                        costYawRateLimit += yawRateCost;
                        gradDelta += yawRatePenalty * dt * penaD * 2.0 * dpsi * invDtSqr;
                        gradT(i) += yawRatePenalty *
                                    (pena - 2.0 * penaD * dpsi * dpsi / (dt * dt));
                    }
                }

                gradPsi(i) -= gradDelta;
                gradPsi(i + 1) += gradDelta;
            }
            return;
        }

        static inline void attachSeamFunctional(const bool hasSeam,
                                                const int seamIdx,
                                                const Eigen::Vector3d &seamPos,
                                                const double seamYaw,
                                                const double seamPosWeight,
                                                const double seamYawWeight,
                                                const Eigen::Matrix3Xd &points,
                                                const Eigen::VectorXd &psi,
                                                double &cost,
                                                double &costSeamPos,
                                                double &costSeamYaw,
                                                Eigen::Matrix3Xd &gradP,
                                                Eigen::VectorXd &gradPsi)
        {
            if (!hasSeam)
            {
                return;
            }

            if (seamIdx <= 0 || seamIdx >= psi.size() - 1)
            {
                return;
            }

            const int pointIdx = seamIdx - 1;
            if (pointIdx >= 0 && pointIdx < points.cols() && seamPosWeight > 0.0)
            {
                const Eigen::Vector3d delta = points.col(pointIdx) - seamPos;
                const double seamPosCost = seamPosWeight * delta.squaredNorm();
                cost += seamPosCost;
                costSeamPos += seamPosCost;
                gradP.col(pointIdx) += 2.0 * seamPosWeight * delta;
            }

            if (seamYawWeight > 0.0)
            {
                const double yawErr = std::remainder(psi(seamIdx) - seamYaw, 2.0 * M_PI);
                const double seamYawCost = seamYawWeight * yawErr * yawErr;
                cost += seamYawCost;
                costSeamYaw += seamYawCost;
                gradPsi(seamIdx) += 2.0 * seamYawWeight * yawErr;
            }
        }

        static inline void attachObservationFunctional(const bool hasObservation,
                                                       const int observationIdx,
                                                       const double observationYaw,
                                                       const double observationYawWeight,
                                                       const Eigen::VectorXd &psi,
                                                       double &cost,
                                                       double &costObservationYaw,
                                                       Eigen::VectorXd &gradPsi)
        {
            if (!hasObservation || observationYawWeight <= 0.0)
            {
                return;
            }

            if (observationIdx <= 0 || observationIdx >= psi.size() - 1)
            {
                return;
            }

            const double yawErr = std::remainder(psi(observationIdx) - observationYaw, 2.0 * M_PI);
            const double obsYawCost = observationYawWeight * yawErr * yawErr;
            cost += obsYawCost;
            costObservationYaw += obsYawCost;
            gradPsi(observationIdx) += 2.0 * observationYawWeight * yawErr;
        }

        static inline double costFunctional(void *ptr,
                                            const Eigen::VectorXd &x,
                                            Eigen::VectorXd &g)
        {
            GCOPTER_PolytopeSFC &obj = *(GCOPTER_PolytopeSFC *)ptr;
            const int dimTau = obj.temporalDim;
            const int dimXi = obj.spatialDim;
            const int dimPsi = std::max(obj.temporalDim - 1, 0);
            const double weightT = obj.rho;
            Eigen::Map<const Eigen::VectorXd> tau(x.data(), dimTau);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + dimTau, dimXi);
            Eigen::Map<Eigen::VectorXd> gradTau(g.data(), dimTau);
            Eigen::Map<Eigen::VectorXd> gradXi(g.data() + dimTau, dimXi);

            forwardT(tau, obj.times);
            forwardP(xi, obj.vPolyIdx, obj.vPolytopes, obj.points);
            if (dimPsi > 0)
            {
                forwardYaw(x.segment(dimTau + dimXi, dimPsi),
                           obj.startYaw,
                           obj.goalYaw,
                           obj.yawKnots);
            }
            else
            {
                obj.yawKnots.resize(2);
                obj.yawKnots(0) = obj.startYaw;
                obj.yawKnots(1) = obj.goalYaw;
            }

            double cost;
            obj.lastCostPos = 0.0;
            obj.lastCostVel = 0.0;
            obj.lastCostOmg = 0.0;
            obj.lastCostTheta = 0.0;
            obj.lastCostThrust = 0.0;
            obj.lastCostYawSmooth = 0.0;
            obj.lastCostYawRateLimit = 0.0;
            obj.lastCostSeamPos = 0.0;
            obj.lastCostSeamYaw = 0.0;
            obj.lastCostObservationYaw = 0.0;
            obj.minco.setParameters(obj.points, obj.times);
            obj.minco.getEnergy(cost);
            cost *= obj.jerkWeight;
            obj.lastCostEnergy = cost;
            obj.minco.getEnergyPartialGradByCoeffs(obj.partialGradByCoeffs);
            obj.minco.getEnergyPartialGradByTimes(obj.partialGradByTimes);
            obj.partialGradByCoeffs *= obj.jerkWeight;
            obj.partialGradByTimes *= obj.jerkWeight;

            attachPenaltyFunctional(obj.times, obj.minco.getCoeffs(),
                                    obj.hPolyIdx, obj.hPolytopes,
                                    obj.smoothEps, obj.integralRes,
                                    obj.magnitudeBd, obj.penaltyWt, obj.flatmap,
                                    cost,
                                    obj.lastCostPos,
                                    obj.lastCostVel,
                                    obj.lastCostOmg,
                                    obj.lastCostTheta,
                                    obj.lastCostThrust,
                                    obj.partialGradByTimes, obj.partialGradByCoeffs);
            attachYawFunctional(obj.times, obj.yawKnots,
                                obj.yawSmoothWeight,
                                obj.maxYawRate,
                                obj.yawRatePenaltyWeight,
                                obj.smoothEps,
                                cost,
                                obj.lastCostYawSmooth,
                                obj.lastCostYawRateLimit,
                                obj.partialGradByTimes, obj.gradByYawKnots);

            obj.minco.propogateGrad(obj.partialGradByCoeffs, obj.partialGradByTimes,
                                    obj.gradByPoints, obj.gradByTimes);
            attachSeamFunctional(obj.hasSeamAnchor,
                                 obj.seamPieceIndex,
                                 obj.seamAnchorPosition,
                                 obj.seamAnchorYaw,
                                 obj.seamAnchorPositionWeight,
                                 obj.seamAnchorYawWeight,
                                 obj.points,
                                 obj.yawKnots,
                                 cost,
                                 obj.lastCostSeamPos,
                                 obj.lastCostSeamYaw,
                                 obj.gradByPoints,
                                 obj.gradByYawKnots);
            attachObservationFunctional(obj.hasObservationCell,
                                        obj.observationKnotIndex,
                                        obj.observationYaw,
                                        obj.observationYawWeight,
                                        obj.yawKnots,
                                        cost,
                                        obj.lastCostObservationYaw,
                                        obj.gradByYawKnots);

            obj.lastCostTime = weightT * obj.times.sum();
            cost += obj.lastCostTime;
            obj.gradByTimes.array() += weightT;

            backwardGradT(tau, obj.gradByTimes, gradTau);
            backwardGradP(xi, obj.vPolyIdx, obj.vPolytopes, obj.gradByPoints, gradXi);
            if (dimPsi > 0)
            {
                g.segment(dimTau + dimXi, dimPsi) = obj.gradByYawKnots.segment(1, dimPsi);
            }
            normRetrictionLayer(xi, obj.vPolyIdx, obj.vPolytopes, cost, gradXi);

            return cost;
        }

        static inline double costDistance(void *ptr,
                                          const Eigen::VectorXd &xi,
                                          Eigen::VectorXd &gradXi)
        {
            void **dataPtrs = (void **)ptr;
            const double &dEps = *((const double *)(dataPtrs[0]));
            const Eigen::Vector3d &ini = *((const Eigen::Vector3d *)(dataPtrs[1]));
            const Eigen::Vector3d &fin = *((const Eigen::Vector3d *)(dataPtrs[2]));
            const PolyhedraV &vPolys = *((PolyhedraV *)(dataPtrs[3]));

            double cost = 0.0;
            const int overlaps = vPolys.size() / 2;

            Eigen::Matrix3Xd gradP = Eigen::Matrix3Xd::Zero(3, overlaps);
            Eigen::Vector3d a, b, d;
            Eigen::VectorXd r;
            double smoothedDistance;
            for (int i = 0, j = 0, k = 0; i <= overlaps; i++, j += k)
            {
                a = i == 0 ? ini : b;
                if (i < overlaps)
                {
                    k = vPolys[2 * i + 1].cols();
                    Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                    r = q.normalized().head(k - 1);
                    b = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                        vPolys[2 * i + 1].col(0);
                }
                else
                {
                    b = fin;
                }

                d = b - a;
                smoothedDistance = sqrt(d.squaredNorm() + dEps);
                cost += smoothedDistance;

                if (i < overlaps)
                {
                    gradP.col(i) += d / smoothedDistance;
                }
                if (i > 0)
                {
                    gradP.col(i - 1) -= d / smoothedDistance;
                }
            }

            Eigen::VectorXd unitQ;
            double sqrNormQ, invNormQ, sqrNormViolation, c, dc;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                Eigen::Map<Eigen::VectorXd> gradQ(gradXi.data() + j, k);
                sqrNormQ = q.squaredNorm();
                invNormQ = 1.0 / sqrt(sqrNormQ);
                unitQ = q * invNormQ;
                gradQ.head(k - 1) = (vPolys[2 * i + 1].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradQ = (gradQ - unitQ * unitQ.dot(gradQ)).eval() * invNormQ;

                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradQ += dc * 2.0 * q;
                }
            }

            return cost;
        }

        static inline void getShortestPath(const Eigen::Vector3d &ini,
                                           const Eigen::Vector3d &fin,
                                           const PolyhedraV &vPolys,
                                           const double &smoothD,
                                           Eigen::Matrix3Xd &path)
        {
            const int overlaps = vPolys.size() / 2;
            if (overlaps <= 0)
            {
                path.resize(3, 2);
                path.col(0) = ini;
                path.col(1) = fin;
                return;
            }
            Eigen::VectorXi vSizes(overlaps);
            for (int i = 0; i < overlaps; i++)
            {
                vSizes(i) = vPolys[2 * i + 1].cols();
                if (vSizes(i) <= 1)
                {
                    path.resize(3, 2);
                    path.col(0) = ini;
                    path.col(1) = fin;
                    return;
                }
            }
            Eigen::VectorXd xi(vSizes.sum());
            for (int i = 0, j = 0; i < overlaps; i++)
            {
                xi.segment(j, vSizes(i)).setConstant(sqrt(1.0 / vSizes(i)));
                j += vSizes(i);
            }

            double minDistance;
            void *dataPtrs[4];
            dataPtrs[0] = (void *)(&smoothD);
            dataPtrs[1] = (void *)(&ini);
            dataPtrs[2] = (void *)(&fin);
            dataPtrs[3] = (void *)(&vPolys);
            lbfgs::lbfgs_parameter_t shortest_path_params;
            shortest_path_params.past = 3;
            shortest_path_params.delta = 1.0e-3;
            shortest_path_params.g_epsilon = 1.0e-5;

            lbfgs::lbfgs_optimize(xi,
                                  minDistance,
                                  &GCOPTER_PolytopeSFC::costDistance,
                                  nullptr,
                                  nullptr,
                                  dataPtrs,
                                  shortest_path_params);

            path.resize(3, overlaps + 2);
            path.leftCols<1>() = ini;
            path.rightCols<1>() = fin;
            Eigen::VectorXd r;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                r = q.normalized().head(k - 1);
                path.col(i + 1) = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                                  vPolys[2 * i + 1].col(0);
            }

            return;
        }

        static inline bool getSeamAwareInitialPath(const Eigen::Vector3d &ini,
                                                   const Eigen::Vector3d &seam,
                                                   const Eigen::Vector3d &fin,
                                                   const PolyhedraH &hPolys,
                                                   const int splitPolyIndex,
                                                   const double &smoothD,
                                                   Eigen::Matrix3Xd &path)
        {
            if (splitPolyIndex <= 0 || splitPolyIndex >= static_cast<int>(hPolys.size()))
            {
                return false;
            }

            PolyhedraH headH(hPolys.begin(), hPolys.begin() + splitPolyIndex);
            PolyhedraH tailH(hPolys.begin() + splitPolyIndex, hPolys.end());
            if (headH.empty() || tailH.empty())
            {
                return false;
            }

            PolyhedraV headV;
            PolyhedraV tailV;
            if (!processCorridor(headH, headV) || !processCorridor(tailH, tailV))
            {
                return false;
            }

            Eigen::Matrix3Xd headPath;
            Eigen::Matrix3Xd tailPath;
            getShortestPath(ini, seam, headV, smoothD, headPath);
            getShortestPath(seam, fin, tailV, smoothD, tailPath);

            if (headPath.cols() != splitPolyIndex + 1 ||
                tailPath.cols() != static_cast<int>(tailH.size()) + 1)
            {
                return false;
            }

            path.resize(3, headPath.cols() + tailPath.cols() - 1);
            path.leftCols(headPath.cols()) = headPath;
            path.rightCols(tailPath.cols() - 1) = tailPath.rightCols(tailPath.cols() - 1);
            return true;
        }

        static inline bool processCorridor(const PolyhedraH &hPs,
                                           PolyhedraV &vPs)
        {
            if (hPs.empty())
            {
                return false;
            }

            const int sizeCorridor = hPs.size() - 1;

            vPs.clear();
            vPs.reserve(2 * sizeCorridor + 1);

            int nv;
            PolyhedronH curIH;
            PolyhedronV curIV, curIOB;
            for (int i = 0; i < sizeCorridor; i++)
            {
                if (!geo_utils::enumerateVs(hPs[i], curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                if (nv <= 0)
                {
                    return false;
                }
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                if (nv > 1)
                {
                    curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                }
                vPs.push_back(curIOB);

                curIH.resize(hPs[i].rows() + hPs[i + 1].rows(), 4);
                curIH.topRows(hPs[i].rows()) = hPs[i];
                curIH.bottomRows(hPs[i + 1].rows()) = hPs[i + 1];
                if (!geo_utils::enumerateVs(curIH, curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                if (nv <= 1)
                {
                    return false;
                }
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                if (nv > 1)
                {
                    curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                }
                vPs.push_back(curIOB);
            }

            if (!geo_utils::enumerateVs(hPs.back(), curIV))
            {
                return false;
            }
            nv = curIV.cols();
            if (nv <= 0)
            {
                return false;
            }
            curIOB.resize(3, nv);
            curIOB.col(0) = curIV.col(0);
            if (nv > 1)
            {
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
            }
            vPs.push_back(curIOB);

            return true;
        }

        static inline void setInitial(const Eigen::Matrix3Xd &path,
                                      const double &speed,
                                      const double &maxAcc,
                                      const double &timeMargin,
                                      const Eigen::VectorXi &intervalNs,
                                      Eigen::Matrix3Xd &innerPoints,
                                      Eigen::VectorXd &timeAlloc)
        {
            const int sizeM = intervalNs.size();
            const int sizeN = intervalNs.sum();
            innerPoints.resize(3, sizeN - 1);
            timeAlloc.resize(sizeN);

            Eigen::Vector3d a, b, c;
            for (int i = 0, j = 0, k = 0, l; i < sizeM; i++)
            {
                l = intervalNs(i);
                a = path.col(i);
                b = path.col(i + 1);
                c = (b - a) / l;
                const double pieceLen = c.norm();
                const double timeByVel = pieceLen / std::max(speed, 1.0e-6);
                const double timeByAcc = std::sqrt(2.0 * pieceLen / std::max(maxAcc, 1.0e-6));
                const double pieceTime =
                    std::max(std::max(timeByVel, timeByAcc) * std::max(timeMargin, 1.0), 1.0e-3);
                timeAlloc.segment(j, l).setConstant(pieceTime);
                j += l;
                for (int m = 0; m < l; m++)
                {
                    if (i > 0 || m > 0)
                    {
                        innerPoints.col(k++) = a + c * m;
                    }
                }
            }
        }

    public:
        // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
        // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
        // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
        //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
        inline bool setup(const double &timeWeight,
                          const double &jerkWeightIn,
                          const Eigen::Matrix3d &initialPVA,
                          const Eigen::Matrix3d &terminalPVA,
                          const PolyhedraH &safeCorridor,
                          const double &lengthPerPiece,
                          const double &smoothingFactor,
                          const int &integralResolution,
                          const Eigen::VectorXd &magnitudeBounds,
                          const Eigen::VectorXd &penaltyWeights,
                          const Eigen::VectorXd &physicalParams,
                          const double &initAllocSpeedRatio,
                          const double &initAllocMaxAcc,
                          const double &initTimeMarginIn,
                          const double &initialYaw,
                          const double &terminalYaw,
                          const double &yawSmoothWeightIn,
                          const double &maxYawRateIn,
                          const double &yawRatePenaltyIn,
                          const SetupOptions &options = SetupOptions())
        {
            rho = timeWeight;
            jerkWeight = std::max(jerkWeightIn, 0.0);
            headPVA = initialPVA;
            tailPVA = terminalPVA;
            startYaw = initialYaw;
            goalYaw = unwrapAngleNear(initialYaw, terminalYaw);
            yawSmoothWeight = yawSmoothWeightIn;
            maxYawRate = maxYawRateIn;
            yawRatePenaltyWeight = yawRatePenaltyIn;
            cellMeta.clear();
            hasSeamAnchor = false;
            seamPieceIndex = -1;
            seamAnchorPosition = Eigen::Vector3d::Zero();
            seamAnchorYaw = 0.0;
            seamAnchorPositionWeight = 0.0;
            seamAnchorYawWeight = 0.0;
            hasObservationCell = false;
            observationCellIndex = -1;
            observationKnotIndex = -1;
            observationYaw = 0.0;
            observationYawWeight = 0.0;

            if (safeCorridor.empty())
            {
                return false;
            }

            hPolytopes = safeCorridor;
            cellMeta = options.cellMeta;
            for (size_t i = 0; i < hPolytopes.size(); i++)
            {
                const Eigen::ArrayXd norms =
                    hPolytopes[i].leftCols<3>().rowwise().norm();
                hPolytopes[i].array().colwise() /= norms;
            }
            if (cellMeta.empty())
            {
                cellMeta.resize(hPolytopes.size());
            }
            else if (cellMeta.size() != hPolytopes.size())
            {
                return false;
            }
            if (!processCorridor(hPolytopes, vPolytopes))
            {
                return false;
            }

            polyN = hPolytopes.size();
            smoothEps = smoothingFactor;
            integralRes = integralResolution;
            magnitudeBd = magnitudeBounds;
            penaltyWt = penaltyWeights;
            physicalPm = physicalParams;
            allocSpeed = std::max(magnitudeBd(0) * initAllocSpeedRatio, 1.0e-3);
            initMaxAcc = std::max(initAllocMaxAcc, 1.0e-3);
            initTimeMargin = std::max(initTimeMarginIn, 1.0);

            if (options.seamAnchor.enabled)
            {
                if (!getSeamAwareInitialPath(headPVA.col(0),
                                             options.seamAnchor.position,
                                             tailPVA.col(0),
                                             hPolytopes,
                                             options.seamAnchor.splitPolyIndex,
                                             smoothEps,
                                             shortPath))
                {
                    return false;
                }
            }
            else
            {
                getShortestPath(headPVA.col(0), tailPVA.col(0),
                                vPolytopes, smoothEps, shortPath);
            }
            if (polyN <= 0 || shortPath.cols() < polyN + 1)
            {
                return false;
            }
            const Eigen::Matrix3Xd deltas = shortPath.rightCols(polyN) - shortPath.leftCols(polyN);
            pieceIdx = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
            if (pieceIdx.size() != polyN)
            {
                return false;
            }
            pieceIdx.array() += 1;
            for (int i = 0; i < polyN; ++i)
            {
                pieceIdx(i) = std::max(pieceIdx(i), std::max(cellMeta[i].minPieces, 1));
            }
            pieceN = pieceIdx.sum();
            if (pieceN <= 0)
            {
                return false;
            }
            if (options.seamAnchor.enabled)
            {
                if (options.seamAnchor.splitPolyIndex <= 0 || options.seamAnchor.splitPolyIndex >= polyN)
                {
                    return false;
                }
                seamPieceIndex = pieceIdx.head(options.seamAnchor.splitPolyIndex).sum();
                if (seamPieceIndex <= 0 || seamPieceIndex >= pieceN)
                {
                    return false;
                }
                hasSeamAnchor = true;
                seamAnchorPosition = options.seamAnchor.position;
                seamAnchorYaw = unwrapAngleNear(startYaw, options.seamAnchor.yaw);
                seamAnchorPositionWeight = options.seamAnchor.positionWeight;
                seamAnchorYawWeight = options.seamAnchor.yawWeight;
            }

            if (options.observationCellIndex >= 0)
            {
                if (options.observationCellIndex >= polyN)
                {
                    return false;
                }
                const int observationPieces = pieceIdx(options.observationCellIndex);
                if (observationPieces < 2)
                {
                    return false;
                }
                hasObservationCell = true;
                observationCellIndex = options.observationCellIndex;
                observationKnotIndex = pieceIdx.head(observationCellIndex).sum() + 1;
                if (observationKnotIndex <= 0 || observationKnotIndex >= pieceN)
                {
                    return false;
                }
                observationYaw = unwrapAngleNear(startYaw, options.observationYaw);
                observationYawWeight = options.observationYawWeight;
            }

            initialYawGuessKnots.resize(pieceN + 1);
            if (hasObservationCell)
            {
                buildInitialYawGuessWithObservation(pieceIdx,
                                                    startYaw,
                                                    observationYaw,
                                                    goalYaw,
                                                    observationCellIndex,
                                                    initialYawGuessKnots);
            }
            else if (hasSeamAnchor)
            {
                buildInitialYawGuessWithSeam(pieceIdx, startYaw, seamAnchorYaw, goalYaw,
                                             seamPieceIndex, initialYawGuessKnots);
            }
            else
            {
                buildInitialYawGuess(shortPath, pieceIdx, startYaw, goalYaw, initialYawGuessKnots);
            }

            temporalDim = pieceN;
            spatialDim = 0;
            vPolyIdx.resize(pieceN - 1);
            hPolyIdx.resize(pieceN);
            for (int i = 0, j = 0, k; i < polyN; i++)
            {
                k = pieceIdx(i);
                for (int l = 0; l < k; l++, j++)
                {
                    if (l < k - 1)
                    {
                        vPolyIdx(j) = 2 * i;
                        spatialDim += vPolytopes[2 * i].cols();
                    }
                    else if (i < polyN - 1)
                    {
                        vPolyIdx(j) = 2 * i + 1;
                        spatialDim += vPolytopes[2 * i + 1].cols();
                    }
                    hPolyIdx(j) = i;
                }
            }

            // Setup for MINCO_S3NU, FlatnessMap, and L-BFGS solver
            minco.setConditions(headPVA, tailPVA, pieceN);
            flatmap.reset(physicalPm(0), physicalPm(1), physicalPm(2),
                          physicalPm(3), physicalPm(4), physicalPm(5));

            // Allocate temp variables
            points.resize(3, pieceN - 1);
            times.resize(pieceN);
            gradByPoints.resize(3, pieceN - 1);
            gradByTimes.resize(pieceN);
            yawKnots.resize(pieceN + 1);
            gradByYawKnots.resize(pieceN + 1);
            partialGradByCoeffs.resize(6 * pieceN, 3);
            partialGradByTimes.resize(pieceN);

            return true;
        }

        inline DebugSnapshot getDebugSnapshot() const
        {
            DebugSnapshot snapshot;
            snapshot.polyNum = polyN;
            snapshot.pieceNum = pieceN;
            snapshot.allocSpeed = allocSpeed;
            snapshot.hasSeamAnchor = hasSeamAnchor;
            snapshot.seamPieceIndex = seamPieceIndex;
            snapshot.seamPosition = seamAnchorPosition;
            snapshot.seamYaw = seamAnchorYaw;
            snapshot.shortPath = shortPath;
            snapshot.pieceIdx = pieceIdx;
            snapshot.vPolyIdx = vPolyIdx;
            snapshot.hPolyIdx = hPolyIdx;
            snapshot.cellTypes.clear();
            snapshot.cellTypes.reserve(cellMeta.size());
            for (const auto &meta : cellMeta)
            {
                snapshot.cellTypes.push_back(static_cast<int>(meta.type));
            }
            snapshot.observationCellIndex = observationCellIndex;
            snapshot.observationKnotIndex = observationKnotIndex;
            snapshot.initialPoints = initialPointsSnapshot;
            snapshot.initialTimes = initialTimesSnapshot;
            snapshot.initialYawKnots = initialYawKnotsSnapshot;
            snapshot.finalPoints = points;
            snapshot.finalTimes = times;
            snapshot.finalYawKnots = yawKnots;
            snapshot.costEnergy = lastCostEnergy;
            snapshot.costTime = lastCostTime;
            snapshot.costPos = lastCostPos;
            snapshot.costVel = lastCostVel;
            snapshot.costOmg = lastCostOmg;
            snapshot.costTheta = lastCostTheta;
            snapshot.costThrust = lastCostThrust;
            snapshot.costYawSmooth = lastCostYawSmooth;
            snapshot.costYawRateLimit = lastCostYawRateLimit;
            snapshot.costSeamPos = lastCostSeamPos;
            snapshot.costSeamYaw = lastCostSeamYaw;
            snapshot.costObservationYaw = lastCostObservationYaw;
            return snapshot;
        }

        inline double optimize(Trajectory<5> &traj,
                               Eigen::VectorXd &yawJunctions,
                               const double &relCostTol)
        {
            const int yawDim = std::max(temporalDim - 1, 0);
            Eigen::VectorXd x(temporalDim + spatialDim + yawDim);
            Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

            setInitial(shortPath, allocSpeed, initMaxAcc, initTimeMargin, pieceIdx, points, times);
            backwardT(times, tau);
            backwardP(points, vPolyIdx, vPolytopes, xi);
            setInitialYaw(startYaw, goalYaw, initialYawGuessKnots, yawKnots);
            initialPointsSnapshot = points;
            initialTimesSnapshot = times;
            initialYawKnotsSnapshot = yawKnots;
            if (yawDim > 0)
            {
                x.segment(temporalDim + spatialDim, yawDim) =
                    yawKnots.segment(1, yawDim);
            }

            double minCostFunctional;
            lbfgs_params.mem_size = 256;
            lbfgs_params.past = 3;
            lbfgs_params.min_step = 1.0e-32;
            lbfgs_params.g_epsilon = 0.0;
            lbfgs_params.delta = relCostTol;

            int ret = lbfgs::lbfgs_optimize(x,
                                            minCostFunctional,
                                            &GCOPTER_PolytopeSFC::costFunctional,
                                            nullptr,
                                            nullptr,
                                            this,
                                            lbfgs_params);

            if (ret >= 0)
            {
                forwardT(tau, times);
                forwardP(xi, vPolyIdx, vPolytopes, points);
                if (yawDim > 0)
                {
                    forwardYaw(x.segment(temporalDim + spatialDim, yawDim),
                               startYaw, goalYaw, yawKnots);
                }
                else
                {
                    yawKnots.resize(2);
                    yawKnots(0) = startYaw;
                    yawKnots(1) = goalYaw;
                }
                minco.setParameters(points, times);
                minco.getTrajectory(traj);
                yawJunctions = yawKnots;
            }
            else
            {
                traj.clear();
                yawJunctions.resize(0);
                minCostFunctional = INFINITY;
                std::cout << "Optimization Failed: "
                          << lbfgs::lbfgs_strerror(ret)
                          << std::endl;
            }

            return minCostFunctional;
        }
    };

}

#endif
