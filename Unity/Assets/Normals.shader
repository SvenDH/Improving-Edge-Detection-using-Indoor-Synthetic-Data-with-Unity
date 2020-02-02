Shader "Normals"
{
	SubShader
	{
		Tags{ "RenderType" = "Opaque"}
		Pass
	{
	CGPROGRAM
	#pragma vertex vert
	#pragma fragment frag
	#include "UnityCG.cginc"

	struct v2f {
		half3 viewNormal : TEXCOORD0;
		float4 pos : SV_POSITION;
	};

	v2f vert(float4 vertex : POSITION, float3 normal : NORMAL)
	{
		v2f o;
		o.pos = UnityObjectToClipPos(vertex);
		float3 worldNormal = UnityObjectToWorldNormal(normal);
		o.viewNormal = normalize(mul((float3x3)UNITY_MATRIX_V, worldNormal));

		return o;
	}

	fixed4 frag(v2f i) : SV_Target
	{
		fixed4 c = 0;

		c.rgb = i.viewNormal*0.5 + 0.5;
		return c;
	}
	ENDCG
}
	}
}
