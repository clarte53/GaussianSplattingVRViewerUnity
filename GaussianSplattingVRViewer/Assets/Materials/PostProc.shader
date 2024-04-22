Shader "Debug/PostProc"
{
    Properties
    {
        _MainTex ("Main Texture", 2D) = "white" {}
        [MaterialToggle] _isWithColor("isWithColor", Float) = 0
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

            float _isWithColor;
            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            UNITY_DECLARE_SCREENSPACE_TEXTURE(_MainTex);
            half4 _MainTex_ST;

            float4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv).r;
                depth = Linear01Depth(depth);
                float4 col = UNITY_SAMPLE_SCREENSPACE_TEXTURE(_MainTex, UnityStereoScreenSpaceUVAdjust(i.uv, _MainTex_ST));
                
                if (_isWithColor > 0.5) {
                    col = col * depth;
                } else {
                    col = depth;
                }
                return col;
            }
            ENDCG
        }
    }
}
