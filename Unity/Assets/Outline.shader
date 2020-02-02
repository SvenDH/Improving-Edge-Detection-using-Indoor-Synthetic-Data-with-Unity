Shader "Outline"
{
	Properties
	{
		_MainTex("Texture", 2D) = "black" {}
		_DepthTex("Texture", 2D) = "black" {}
		_ResX("Resolution X", Float) = 1024
		_ResY("Resolution Y", Float) = 1024
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
			float4 _MainTex_ST;
			float _ResX;
			float _ResY;

			v2f vert(appdata v) {
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				return o;
			}

			float2 SobelDir(fixed3 textures[9]) {
				float3x3 KERNEL_SOBELX = float3x3(
					-1.0, 0.0, 1.0,
					-2.0, 0.0, 2.0,
					-1.0, 0.0, 1.0
					);
				float3x3 KERNEL_SOBELY = float3x3(
					1.0, 2.0, 1.0,
					0.0, 0.0, 0.0,
					-1.0, -2.0, -1.0
					);

				float mGx = 0.0;
				float mGy = 0.0;

				for (int i = 0, k = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++, k++)
					{
						float Gx = KERNEL_SOBELX[i][j] * Grayscale(textures[k]);
						float Gy = KERNEL_SOBELY[i][j] * Grayscale(textures[k]);

						mGx += Gx;
						mGy += Gy;
					}
				}

				return float2(mGx, mGy);
			}

			fixed4 frag(v2f i) : SV_Target {
				// sample the texture
				float2 offsets[9];
				GetOffsets3x3(_ResX, _ResY, offsets);

				fixed3 textures[9];
				for (int j = 0; j < 9; j++) {
					textures[j] = tex2D(_MainTex, i.uv + offsets[j]).rgb;
				}

				fixed3 depth[9];
				for (int j = 0; j < 9; j++) {
					depth[j] = tex2D(_DepthTex, i.uv + offsets[j]).rgb;
				}

				if ((textures[4].r != textures[1].r && depth[4].r < depth[1].r) ||
					(textures[4].r != textures[3].r && depth[4].r < depth[3].r) ||
					(textures[4].r != textures[5].r && depth[4].r < depth[5].r) ||
					(textures[4].r != textures[7].r && depth[4].r < depth[7].r))
				{
					return ApplySobel(textures);
				}
				return float4(0, 0, 0, 1);
			}
			ENDCG
		}
	}
}
