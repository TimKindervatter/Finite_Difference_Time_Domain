import pytest
import numpy as np
from FDTD_engine import FDTD_engine

def test_FDTD_engine():
    reflected_fourier = np.array([-8.78844504e-14+0.00000000e+00j, -5.04256061e-11-1.00440315e-10j,
       -1.70881682e-10-1.29485280e-10j, -2.92535173e-10-6.03765915e-11j,
       -3.53042970e-10+8.14622760e-11j, -3.24465875e-10+2.43507568e-10j,
       -2.13580573e-10+3.73688438e-10j, -5.06852237e-11+4.34852883e-10j,
        1.21269668e-10+4.10799224e-10j,  2.57186363e-10+3.08816316e-10j,
        3.20479214e-10+1.58758254e-10j,  2.94237198e-10+8.94848158e-12j,
        1.94051954e-10-8.52147718e-11j,  7.30767705e-11-8.66892505e-11j,
        1.43759677e-12-5.64536376e-12j,  2.30088919e-11+9.81340619e-11j,
        1.24867897e-10+1.51912399e-10j,  2.49045875e-10+1.16347913e-10j,
        3.32892123e-10+1.95841974e-12j,  3.40325196e-10-1.48658416e-10j,
        2.67578116e-10-2.86766234e-10j,  1.35155213e-10-3.73237025e-10j,
       -2.14756760e-11-3.85823619e-10j, -1.61077637e-10-3.22912829e-10j,
       -2.46261301e-10-2.04140126e-10j, -2.53637909e-10-6.83432521e-11j,
       -1.86348586e-10+3.40000817e-11j, -8.27379290e-11+6.11973286e-11j,
       -4.32220967e-12+8.38521481e-12j,  2.30635337e-13-8.23675002e-11j,
       -7.21818909e-11-1.47309063e-10j, -1.80354907e-10-1.42458851e-10j,
       -2.69969028e-10-6.50745793e-11j, -3.03428484e-10+5.57929271e-11j,
       -2.69312428e-10+1.80485405e-10j, -1.77978893e-10+2.73557651e-10j,
       -5.46260721e-11+3.11221989e-10j,  6.78354658e-11+2.85591737e-10j,
        1.56497142e-10+2.06351840e-10j,  1.86667234e-10+1.00362964e-10j,
        1.52507490e-10+7.61039982e-12j,  7.68279260e-11-3.30780905e-11j,
        6.37637299e-12-7.51028083e-12j, -1.42371707e-11+6.01933138e-11j,
        2.81009733e-11+1.21698206e-10j,  1.09229206e-10+1.36633819e-10j,
        1.88336342e-10+9.41704101e-11j,  2.32501098e-10+1.04056970e-11j,
        2.27211049e-10-8.66174606e-11j,  1.75019434e-10-1.69027792e-10j,
        9.11093416e-11-2.15694133e-10j, -1.57840623e-12-2.16206489e-10j,
       -7.78431969e-11-1.72950881e-10j, -1.16149452e-10-1.01709692e-10j,
       -1.06576790e-10-3.01365682e-11j, -5.96209055e-11+1.09255778e-11j,
       -6.16065310e-12+4.61126057e-12j,  1.89832565e-11-3.86790130e-11j,
       -3.19481552e-13-8.71120430e-11j, -5.30187726e-11-1.09254239e-10j,
       -1.12453928e-10-9.20073236e-11j, -1.53905879e-10-4.20051155e-11j,
       -1.63865029e-10+2.35498552e-11j, -1.40531684e-10+8.55218846e-11j,
       -9.14429218e-11+1.27873426e-10j, -3.05568548e-11+1.40861749e-10j,
        2.52229124e-11+1.22963361e-10j,  5.98565864e-11+8.18438501e-11j,
        6.34483051e-11+3.42523240e-11j,  3.89365435e-11+1.45639892e-12j,
        4.09991722e-12-1.84937418e-12j, -1.75025209e-11+2.19585112e-11j,
       -1.17736523e-11+5.45909838e-11j,  1.80222264e-11+7.49017431e-11j,
        5.66883440e-11+7.16149775e-11j,  8.81109886e-11+4.60804436e-11j,
        1.02065576e-10+7.42612383e-12j,  9.53796717e-11-3.28250824e-11j,
        7.09213253e-11-6.41138199e-11j,  3.60877056e-11-7.90736063e-11j,
        8.61128221e-13-7.49988396e-11j, -2.43703101e-11-5.47611497e-11j,
       -3.19511617e-11-2.72671912e-11j, -2.13953858e-11-5.40716829e-12j,
       -1.65354026e-12+3.51489792e-13j,  1.32763622e-11-1.09635830e-11j,
        1.34417208e-11-3.01341401e-11j, -1.29611994e-12-4.47082310e-11j,
       -2.33425514e-11-4.68577785e-11j, -4.34928652e-11-3.58665871e-11j,
       -5.52006948e-11-1.59050425e-11j, -5.56563179e-11+6.91613420e-12j,
       -4.54884919e-11+2.65044887e-11j, -2.80976634e-11+3.81470570e-11j,
       -8.69747349e-12+3.94649866e-11j,  6.79046376e-12+3.10899334e-11j,
        1.34169347e-11+1.71922534e-11j,  9.82269993e-12+4.71320036e-12j,
       -3.01471626e-14+1.44049546e-14j, -8.75284356e-12+4.70984915e-12j])

    transmitted_fourier = np.array([ 8.86203370e-10+0.00000000e+00j,  7.85248219e-10-3.94647090e-10j,
        5.17888103e-10-6.85330657e-10j,  1.67091916e-10-8.15311264e-10j,
       -1.82197876e-10-7.83967110e-10j, -4.70588833e-10-6.24109288e-10j,
       -6.65742213e-10-3.77921192e-10j, -7.53221234e-10-8.53598214e-11j,
       -7.27807318e-10+2.17569825e-10j, -5.89448192e-10+4.94994059e-10j,
       -3.45841258e-10+7.06608001e-10j, -2.05228890e-11+8.07419504e-10j,
        3.36736209e-10+7.56436793e-10j,  6.46721563e-10+5.38724047e-10j,
        8.21500595e-10+1.91469691e-10j,  8.08084931e-10-1.95213708e-10j,
        6.19718594e-10-5.17426828e-10j,  3.23335615e-10-7.06373558e-10j,
       -1.52695754e-12-7.45885101e-10j, -2.93442919e-10-6.56362024e-10j,
       -5.15314883e-10-4.71971539e-10j, -6.47815709e-10-2.27613780e-10j,
       -6.81353902e-10+4.46052287e-11j, -6.11240875e-10+3.12884209e-10j,
       -4.38626366e-10+5.41251365e-10j, -1.76992447e-10+6.87785915e-10j,
        1.37870701e-10+7.09904213e-10j,  4.41420183e-10+5.81950774e-10j,
        6.53009985e-10+3.20490062e-10j,  7.12343431e-10-7.28669615e-12j,
        6.12858584e-10-3.11038379e-10j,  3.99770490e-10-5.21067196e-10j,
        1.38015248e-10-6.10605388e-10j, -1.17058193e-10-5.87374363e-10j,
       -3.29052018e-10-4.74845582e-10j, -4.77027410e-10-2.99710361e-10j,
       -5.49112688e-10-8.73248143e-11j, -5.37784595e-10+1.37238111e-10j,
       -4.39501957e-10+3.45499843e-10j, -2.59185052e-10+5.03088775e-10j,
       -1.84684070e-11+5.72104118e-10j,  2.36736808e-10+5.23333162e-10j,
        4.41905811e-10+3.57403178e-10j,  5.40463916e-10+1.16886528e-10j,
        5.13741471e-10-1.29657515e-10j,  3.86216414e-10-3.21858584e-10j,
        2.04004177e-10-4.29758794e-10j,  1.06854061e-11-4.51642394e-10j,
       -1.62833697e-10-4.00707447e-10j, -2.97409064e-10-2.94667249e-10j,
       -3.81309634e-10-1.51170259e-10j, -4.06328896e-10+1.22429216e-11j,
       -3.66737378e-10+1.75555221e-10j, -2.61918852e-10+3.13844248e-10j,
       -1.02336946e-10+3.97727319e-10j,  8.35807517e-11+4.00746113e-10j,
        2.50516549e-10+3.14840277e-10j,  3.53228847e-10+1.62373059e-10j,
        3.69707007e-10-1.11250745e-11j,  3.09312272e-10-1.60355298e-10j,
        2.00434359e-10-2.58697219e-10j,  7.31168237e-11-2.99715916e-10j,
       -4.97799641e-11-2.88969624e-10j, -1.53208522e-10-2.36515610e-10j,
       -2.27426637e-10-1.53101928e-10j, -2.65248527e-10-4.95668209e-11j,
       -2.60890707e-10+6.16113223e-11j, -2.11277238e-10+1.64401256e-10j,
       -1.19852142e-10+2.38750153e-10j, -1.83416934e-12+2.64457078e-10j,
        1.14687627e-10+2.31199890e-10j,  1.98305479e-10+1.48156529e-10j,
        2.29969230e-10+4.18586410e-11j,  2.10697019e-10-5.79812388e-11j,
        1.55563625e-10-1.31580759e-10j,  8.27307485e-11-1.71810787e-10j,
        6.98695268e-12-1.79845498e-10j, -6.14025160e-11-1.60448721e-10j,
       -1.15490121e-10-1.19285901e-10j, -1.49960248e-10-6.22267669e-11j,
       -1.60205848e-10+3.85273409e-12j, -1.42856186e-10+6.97894382e-11j,
       -9.79537852e-11+1.23443588e-10j, -3.24433343e-11+1.51449775e-10j,
        3.82279458e-11+1.45044848e-10j,  9.48934642e-11+1.06614859e-10j,
        1.24089859e-10+4.97725683e-11j,  1.23742772e-10-8.32024736e-12j,
        1.00703578e-10-5.50697968e-11j,  6.46143408e-11-8.48741600e-11j,
        2.38649953e-11-9.71229323e-11j, -1.54339243e-11-9.35969172e-11j,
       -4.89825178e-11-7.68341445e-11j, -7.33853672e-11-4.95855556e-11j,
       -8.55512544e-11-1.51782436e-11j, -8.28545043e-11+2.17325766e-11j,
       -6.42601776e-11+5.45874348e-11j, -3.23864734e-11+7.55954788e-11j,
        5.18269598e-12+7.88148206e-11j,  3.80635477e-11+6.40492393e-11j])

    source_fourier = np.array([ 8.86226925e-10+0.00000000e+00j,  8.69992602e-10-1.67677330e-10j,
        8.21916572e-10-3.29045760e-10j,  7.43854234e-10-4.78046089e-10j,
        6.38814259e-10-6.09108166e-10j,  5.10836834e-10-7.17369958e-10j,
        3.64830483e-10-7.98867326e-10j,  2.06374368e-10-8.50686731e-10j,
        4.14944024e-11-8.71074705e-10j, -1.23577414e-10-8.59499792e-10j,
       -2.82650388e-10-8.16664664e-10j, -4.29818969e-10-7.44468293e-10j,
       -5.59692970e-10-6.45920120e-10j, -6.67604788e-10-5.25010209e-10j,
       -7.49785298e-10-3.86541187e-10j, -8.03501474e-10-2.35929320e-10j,
       -8.27150530e-10-7.89832991e-11j, -8.20307238e-10+7.83298439e-11j,
       -7.83723106e-10+2.30121867e-10j, -7.19278163e-10+3.70813665e-10j,
       -6.29888065e-10+4.95349450e-10j, -5.19371089e-10+5.99386188e-10j,
       -3.92281186e-10+6.79450944e-10j, -2.53714606e-10+7.33060211e-10j,
       -1.09098547e-10+7.58797062e-10j,  3.60290956e-11+7.56343796e-10j,
        1.76239635e-10+7.26469669e-10j,  3.06424107e-10+6.70975201e-10j,
        4.21985956e-10+5.92596357e-10j,  5.19008387e-10+4.94873497e-10j,
        5.94390138e-10+3.81991347e-10j,  6.45944902e-10+2.58597330e-10j,
        6.72461275e-10+1.29606288e-10j,  6.73721866e-10-2.66573214e-25j,
        6.50481940e-10-1.25370118e-10j,  6.04409669e-10-2.41969131e-10j,
        5.37991590e-10-3.45746201e-10j,  4.54408275e-10-4.33277415e-10j,
        3.57386231e-10-5.01878737e-10j,  2.51032940e-10-5.49685464e-10j,
        1.39662346e-10-5.75696029e-10j,  2.76183082e-11-5.79779640e-10j,
       -8.08966694e-11-5.62648694e-10j, -1.81980606e-10-5.25798431e-10j,
       -2.72173026e-10-4.71417510e-10j, -3.48572673e-10-4.02274309e-10j,
       -4.08928058e-10-3.21584579e-10j, -4.51698210e-10-2.32866612e-10j,
       -4.76082381e-10-1.39790400e-10j, -4.82018787e-10-4.60272135e-11j,
       -4.70153787e-10+4.48942433e-11j, -4.41784045e-10+1.29719500e-10j,
       -3.98775285e-10+2.05582948e-10j, -3.43462020e-10+2.70101518e-10j,
       -2.78533280e-10+3.21444540e-10j, -2.06909668e-10+3.58378057e-10j,
       -1.31617233e-10+3.80283020e-10j, -5.56634647e-11+3.87147901e-10j,
        1.80795914e-11+3.79537331e-10j,  8.69806836e-11+3.58539261e-10j,
        1.48739440e-10+3.25693944e-10j,  2.01458313e-10+2.82908616e-10j,
        2.43694402e-10+2.32362143e-10j,  2.74490144e-10+1.76404104e-10j,
        2.93382778e-10+1.17452747e-10j,  3.00393412e-10+5.78960847e-11j,
        2.95997299e-10-2.74716960e-25j,  2.81077661e-10-5.41732789e-11j,
        2.56865911e-10-1.02833599e-10j,  2.24871563e-10-1.44516178e-10j,
        1.86805331e-10-1.78118523e-10j,  1.44498997e-10-2.02920448e-10j,
        9.98255495e-11-2.18587463e-10j,  5.46228443e-11-2.25158431e-10j,
        1.06237057e-11-2.23019029e-10j, -3.06050846e-11-2.12863039e-10j,
       -6.77129965e-11-1.95643855e-10j, -9.96038187e-11-1.72518875e-10j,
       -1.25460850e-10-1.44789539e-10j, -1.44759069e-10-1.13839790e-10j,
       -1.57264690e-10-8.10755825e-11j, -1.63023024e-10-4.78678787e-11j,
       -1.62335978e-10-1.55012064e-11j, -1.55730880e-10+1.48704960e-11j,
       -1.43922538e-10+4.22594700e-11j, -1.27770618e-10+6.58703319e-11j,
       -1.08234438e-10+8.51165026e-11j, -8.63272327e-11+9.96269375e-11j,
       -6.30718387e-11+1.09243629e-10j, -3.94594880e-11+1.14010703e-10j,
       -1.64131866e-11+1.14156220e-10j,  5.24318601e-12+1.10068020e-10j,
        2.48092396e-11+1.02265078e-10j,  4.17254243e-11+9.13659346e-11j,
        5.55832317e-11+7.80557271e-11j,  6.61284184e-11+6.30533196e-11j,
        7.32577067e-11+4.70798694e-11j,  7.70096524e-11+3.08300142e-11j,
        7.75505588e-11+1.49466451e-11j,  7.51564500e-11-5.61395282e-26j])

    conservation = np.array([0.99994685, 0.99998634, 1.00004493, 1.00003471, 0.99997372,
       0.99995663, 1.00000849, 1.00004731, 1.00001253, 0.99995937,
       0.99997287, 1.00003473, 1.0000501 , 0.9999941 , 0.99995183,
       0.99998872, 1.00005021, 1.00004518, 0.99998492, 0.99996422,
       1.00001469, 1.00005708, 1.00002554, 0.99997052, 0.99998039,
       1.00004316, 1.00006359, 1.00001049, 0.9999666 , 1.0000024 ,
       1.00006746, 1.00006711, 1.00000676, 0.99998195, 1.00003031,
       1.00007459, 1.00004429, 0.99998646, 0.99999305, 1.00005735,
       1.00008312, 1.00003385, 0.99999076, 1.00002812, 1.00009741,
       1.00010009, 1.00003758, 1.0000075 , 1.00005251, 1.00009605,
       1.00006401, 1.00000239, 1.00000712, 1.00007471, 1.00010677,
       1.00006287, 1.00002401, 1.00006629, 1.00013985, 1.00014209,
       1.00007372, 1.00003659, 1.00007677, 1.00011633, 1.00007904,
       1.00001332, 1.00001939, 1.00009367, 1.0001337 , 1.00009713,
       1.0000665 , 1.00011709, 1.00019339, 1.00018919, 1.00010954,
       1.0000634 , 1.00009757, 1.00012968, 1.00008377, 1.00001528,
       1.0000285 , 1.0001147 , 1.00016481, 1.00013752, 1.00011919,
       1.00018041, 1.00025504, 1.00023525, 1.00013775, 1.00008147,
       1.00010929, 1.00013074, 1.00007396, 1.00000704, 1.00003651,
       1.00014125, 1.00020314, 1.00018652, 1.00018373, 1.00025523])

    actual_reflected, actual_transmitted, actual_source, actual_conservation = FDTD_engine()

    assert(np.allclose(actual_reflected, reflected_fourier))
    assert(np.allclose(actual_transmitted, transmitted_fourier))
    assert(np.allclose(actual_source, source_fourier))
    assert(np.allclose(actual_conservation, conservation))