Shader "GaussianSplatting/CameraDepthShader"
{
    Properties
    {
        _MainTex ("Main Texture", 2D) = "white" {}
        _Scale("Scale", float) = 1
        _Eye("Eye", int) = 0
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;

                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;

                UNITY_VERTEX_OUTPUT_STEREO
            };

            v2f vert (appdata v)
            {
                v2f o;

                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float _Scale;
            int _Eye;
            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            UNITY_DECLARE_SCREENSPACE_TEXTURE(_MainTex);
            half4 _MainTex_ST;

            float4 frag(v2f i) : SV_Target
            {
                //UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                unity_StereoEyeIndex = _Eye;
                float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv).r;
                depth = LinearEyeDepth(depth);
                depth /= _Scale;
                return depth;
            }
            ENDCG
        }
    }
}
