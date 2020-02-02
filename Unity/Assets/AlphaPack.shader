Shader "AlphaPack"
{
	Properties
	{
		_MainTex("Texture", 2D) = "black" {}
		_AlphaTex("Texture", 2D) = "black" {}
		_DepthTex("Texture", 2D) = "black" {}
		_ResX("Resolution X", Float) = 1024
		_ResY("Resolution Y", Float) = 1024
		_Threshold("Threshold", Float) = 0.5
	}
		SubShader
		{
			Tags { "RenderType" = "Opaque" }

			Cull Off ZWrite Off ZTest Always

			Pass
			{
				CGPROGRAM
				#pragma vertex vert
				#pragma fragment frag

				#include "PostFunctions.cginc"

				struct appdata
				{
					float4 vertex : POSITION;
					float2 uv : TEXCOORD0;
				};

				struct v2f
				{
					float2 uv : TEXCOORD0;
					float4 vertex : SV_POSITION;
				};

				sampler2D _MainTex;
				sampler2D _DepthTex;
				sampler2D _AlphaTex;
				float4 _MainTex_ST;
				float _ResX;
				float _ResY;
				float _Threshold;

				v2f vert(appdata v) {
					v2f o;
					o.vertex = UnityObjectToClipPos(v.vertex);
					o.uv = TRANSFORM_TEX(v.uv, _MainTex);
					return o;
				}

				fixed4 frag(v2f i) : SV_Target {
					fixed4 SobelColor = tex2D(_AlphaTex, i.uv);
					if (SobelColor.r <= _Threshold) {
						discard;
					}
					fixed4 FragColor = tex2D(_MainTex, i.uv);
					//return fixed4(tex2D(_MainTex, i.uv).rgb, 0.0);
					return FragColor;
				}
				ENDCG
			}
		}
}
